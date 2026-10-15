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

import os
import shutil
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
    """llama-server: the path given, $CUE_LLAMA_SERVER, or the one on PATH."""
    for candidate in (path, os.environ.get("CUE_LLAMA_SERVER"), "llama-server"):
        if candidate:
            found = shutil.which(str(candidate))
            if found:
                return found
    raise FileNotFoundError(INSTALL)


def running(url: str, timeout: float = 2.0, api_key: str | None = None) -> bool:
    """Whether an OpenAI-compatible server answers at this base URL (e.g. http://localhost:8090/v1)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        return httpx.get(url.rstrip("/") + "/models", timeout=timeout, headers=headers).status_code == 200
    except httpx.HTTPError:
        return False


class Server:
    """llama-server serving Cue v1 on this machine. ``start()`` downloads the model if needed, launches the server
    and waits until it answers; ``stop()`` (or leaving a ``with`` block) shuts it down. If a server already answers at
    the address, it is used as it is and left running.

    The defaults are the benchmarked setup: 4 parallel slots, 8192 tokens of context shared between them, every layer
    on the GPU (``gpu_layers=0`` for CPU only, which works but is slower)."""

    def __init__(self, *, port: int = 8090, host: str = "127.0.0.1", model: str | Path | None = None,
                 parallel: int = 4, ctx: int = 8192, gpu_layers: int = 999, llama_server: str | Path | None = None,
                 args: tuple[str, ...] = ()) -> None:
        self.port, self.host, self.model = port, host, model
        self.parallel, self.ctx, self.gpu_layers = parallel, ctx, gpu_layers
        self.llama_server, self.args = llama_server, tuple(args)
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.reused = False

    @property
    def url(self) -> str:
        """The base URL to give ``Cue``."""
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}/v1"

    def command(self, model: str | Path, binary: str | None = None) -> list[str]:
        return [binary or find_llama_server(self.llama_server), "-m", str(model), "-c", str(self.ctx),
                "-np", str(self.parallel), "-ngl", str(self.gpu_layers), "--jinja", "--host", self.host,
                "--port", str(self.port), *self.args]

    def start(self, timeout: float = 600.0) -> Server:
        if self.process is not None and self.process.poll() is None:
            return self                                       # already started by us
        if running(self.url):
            self.reused = True
            return self
        binary = find_llama_server(self.llama_server)          # before the download: no 8 GB for nothing to run it
        command = self.command(self.model or download(), binary)
        fd, log = tempfile.mkstemp(prefix="cue-llama-server-", suffix=".log")
        self.log_path = Path(log)
        with os.fdopen(fd, "w") as out:
            self.process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        health = self.url[: -len("/v1")] + "/health"
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server stopped (exit {self.process.returncode}) before it was ready:\n"
                                   f"{self.log_tail()}\nfull log: {self.log_path}")
            try:
                if httpx.get(health, timeout=2.0).status_code == 200:     # 503 while the model loads
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server was not ready after {timeout:.0f} s:\n{self.log_tail()}")

    def log_tail(self, lines: int = 15) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])

    def wait(self) -> int:
        """Block until the server exits (a foreground ``cue serve``)."""
        return self.process.wait() if self.process is not None else 0

    def stop(self) -> None:
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
    call ``stop()`` when done."""
    return Server(**kwargs).start()  # type: ignore[arg-type]
