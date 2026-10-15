"""Get Cue v1 running: download it from Hugging Face once, and serve it with llama.cpp's llama-server.

    from cue import Cue, serve
    with serve() as server:                  # downloads Pype/cue-v1 once (8 GB), starts llama-server, waits for it
        cue = Cue(server.url)
        cue.end_of_turn("Sure, it's nine eight four five,", agent_said="What's the best number to reach you on?")

Or from a terminal: ``cue serve``. llama-server itself is llama.cpp's (``brew install llama.cpp`` on a Mac, or the
prebuilt releases and Docker images at https://github.com/ggml-org/llama.cpp); Cue starts it with the flags Cue v1 was
benchmarked with. Any OpenAI-compatible server that returns logprobs works too: point ``Cue`` at its URL.
"""
from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

REPO = "Pype/cue-v1"
FILE = "cue-v1-q8_0.gguf"

INSTALL = """llama-server (llama.cpp) was not found. Install it, then run again:
  macOS:          brew install llama.cpp
  Linux, Windows: a prebuilt release from https://github.com/ggml-org/llama.cpp/releases
                  (or build it with CUDA for an NVIDIA GPU)
  Docker:         ghcr.io/ggml-org/llama.cpp:server-cuda, with --hf-repo Pype/cue-v1 --hf-file cue-v1-q8_0.gguf
Or tell Cue where it is: CUE_LLAMA_SERVER=/path/to/llama-server, or llama_server= / --llama-server."""


def download(repo: str = REPO, filename: str = FILE, *, revision: str | None = None) -> Path:
    """The Cue v1 model file, downloaded from Hugging Face on first use and cached (~/.cache/huggingface)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, revision=revision))


def find_llama_server(path: str | Path | None = None) -> str:
    """llama-server: the path given, else $CUE_LLAMA_SERVER, else the one on PATH. A path you name must work: it is
    never silently swapped for another llama-server."""
    for candidate, where in ((path, "llama_server / --llama-server"), (os.environ.get("CUE_LLAMA_SERVER"),
                                                                         "$CUE_LLAMA_SERVER")):
        if candidate:
            found = shutil.which(str(candidate))
            if not found:
                raise FileNotFoundError(f"{where} is {candidate}, which is not an executable llama-server")
            return found
    found = shutil.which("llama-server")
    if found:
        return found
    raise FileNotFoundError(INSTALL)


def probe(url: str, timeout: float = 2.0, api_key: str | None = None) -> str:
    """What answers at this base URL: "ready", "loading" (llama-server answers 503 until its model is loaded),
    "needs_key" (401 or 403: it wants an API key), "other" (something that is not an OpenAI-compatible server) or
    "none"."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        status = httpx.get(url.rstrip("/") + "/models", timeout=timeout, headers=headers).status_code
    except (httpx.HTTPError, httpx.InvalidURL):
        return "none"
    return {200: "ready", 503: "loading", 401: "needs_key", 403: "needs_key"}.get(status, "other")


def running(url: str, timeout: float = 2.0, api_key: str | None = None) -> bool:
    """Whether a server is ready to answer at this base URL (e.g. http://localhost:8090/v1)."""
    return probe(url, timeout, api_key) == "ready"


def models(url: str, api_key: str | None = None) -> list[str]:
    """The model ids a server lists (llama-server lists the GGUF it loaded)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        body = httpx.get(url.rstrip("/") + "/models", timeout=2.0, headers=headers).json()
    except Exception:
        return []
    return [str(m.get("id") or m.get("name") or m.get("model")) for m in body.get("data") or body.get("models") or []]


def wait_ready(url: str, timeout: float = 900.0, api_key: str | None = None) -> None:
    """Wait for a server that is loading its model."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = probe(url, api_key=api_key)
        if state == "ready":
            return
        if state not in ("loading", "none"):
            raise RuntimeError(f"the server at {url} stopped loading: it answered {state}")
        time.sleep(1.0)
    raise TimeoutError(f"the server at {url} was still not ready after {timeout:.0f} s")


def _port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


class Server:
    """llama-server serving Cue v1 on this machine. ``start()`` downloads the model if needed, launches the server
    and waits until it answers; ``stop()`` (or leaving a ``with`` block, or Python exiting) shuts it down. If a server
    already answers at the address, or is loading there, it is used as it is and left running.

    The defaults are the benchmarked setup: 4 parallel slots, 8192 tokens of context shared between them, every layer
    on the GPU (``gpu_layers=0`` for CPU only, which works but is slower). ``api_key``: clients must send it (use one
    with ``host="0.0.0.0"``, which serves other machines)."""

    def __init__(self, *, port: int = 8090, host: str = "127.0.0.1", model: str | Path | None = None,
                 parallel: int = 4, ctx: int = 8192, gpu_layers: int = 999, llama_server: str | Path | None = None,
                 api_key: str | None = None, args: tuple[str, ...] = ()) -> None:
        self.port, self.host, self.model = port, host, model
        self.parallel, self.ctx, self.gpu_layers = parallel, ctx, gpu_layers
        self.llama_server, self.api_key, self.args = llama_server, api_key, tuple(args)
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.reused = False

    @property
    def url(self) -> str:
        """The base URL to give ``Cue``."""
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.host, self.host)
        return f"http://{f'[{host}]' if ':' in host else host}:{self.port}/v1"

    def command(self, model: str | Path, binary: str | None = None) -> list[str]:
        key = ["--api-key", self.api_key] if self.api_key else []
        return [binary or find_llama_server(self.llama_server), "-m", str(model), "-c", str(self.ctx),
                "-np", str(self.parallel), "-ngl", str(self.gpu_layers), "--jinja", "--host", self.host,
                "--port", str(self.port), *key, *self.args]

    def start(self, timeout: float = 600.0) -> Server:
        if self.process is not None and self.process.poll() is None:
            return self                                       # already started by us
        state = probe(self.url, api_key=self.api_key)
        if state in ("ready", "loading"):                     # a server is here already: use it, leave it running
            if state == "loading":
                wait_ready(self.url, timeout, self.api_key)
            self.reused = True
            return self
        if state == "needs_key":
            raise RuntimeError(f"a server at {self.url} wants an API key: pass api_key (--api-key), or use another "
                               "port")
        if state == "other" or not _port_free(self.host, self.port):
            raise RuntimeError(f"port {self.port} is in use by something that is not a Cue server: use another port")
        binary = find_llama_server(self.llama_server)          # before the download: no 8 GB for nothing to run it
        try:
            model = self.model or download()
        except Exception as e:
            raise RuntimeError(f"could not download {REPO}/{FILE} from Hugging Face: {e}") from e
        fd, log = tempfile.mkstemp(prefix="cue-llama-server-", suffix=".log")
        self.log_path = Path(log)
        with os.fdopen(fd, "w") as out:
            self.process = subprocess.Popen(self.command(model, binary), stdout=out, stderr=subprocess.STDOUT)
        atexit.register(self.stop)                            # never leave 8 GB loaded behind a crashed program
        health = self.url[: -len("/v1")] + "/health"
        try:
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server stopped (exit {self.process.returncode}) before it was ready:\n"
                                       f"{self.log_tail()}\nfull log: {self.log_path}")
                try:
                    if httpx.get(health, timeout=2.0).status_code == 200:     # 503 while the model loads
                        time.sleep(0.2)
                        if self.process.poll() is None:      # ours answered, not something else on the port
                            return self
                        continue
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            raise TimeoutError(f"llama-server was not ready after {timeout:.0f} s:\n{self.log_tail()}")
        except BaseException:
            self.stop()
            raise

    def log_tail(self, lines: int = 15) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])

    def wait(self) -> int:
        """Block until the server exits (a foreground ``cue serve``). It polls rather than blocking in waitpid, so a
        signal handler can stop the server cleanly."""
        while self.process is not None and self.process.poll() is None:
            time.sleep(0.5)
        return self.process.returncode if self.process is not None else 0

    def stop(self) -> None:
        atexit.unregister(self.stop)
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def __enter__(self) -> Server:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve(**kwargs: object) -> Server:
    """Start Cue v1 on this machine (see ``Server``) and return it once it answers. Use it in a ``with`` block, or
    call ``stop()`` when done (it is also stopped when Python exits)."""
    return Server(**kwargs).start()  # type: ignore[arg-type]
