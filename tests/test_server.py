"""Getting Cue running and asking it: prompts by name, the three questions, the server (download, start, reuse, fail
clearly) and the ``cue`` command. llama-server is played by tests/fake_llama_server.py; nothing is downloaded."""
import asyncio
import os
import socket
import stat
import sys
from pathlib import Path

import pytest

import cue.server as server_mod
from cue import Cue, Server, UNPUNCTUATED, load, prompt
from cue.cli import main
from cue.client import barge_in_inputs, dialogue_inputs, end_of_turn_inputs

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def llama(tmp_path):
    """An executable called llama-server that runs the stand-in, and a model file for it."""
    exe = tmp_path / "llama-server"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / "fake_llama_server.py"}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "cue-v1-q8_0.gguf"
    model.write_bytes(b"GGUF")
    return exe, model


@pytest.fixture
def running_server(llama):
    exe, model = llama
    with Server(port=free_port(), model=model, llama_server=exe) as server:
        yield server


# -- prompts and inputs ----------------------------------------------------------------------------------------
def test_the_prompts_ship_inside_the_package_and_load_by_name(tmp_path, monkeypatch):
    assert {p.stem for p in prompt("barge_in").parent.glob("*.json")} == {
        "end_of_turn_phone", "end_of_turn_dialogue", "barge_in"}
    assert load("end_of_turn_phone").ids == ("DONE", "LIKELY_DONE", "MID_THOUGHT", "HOLDING")
    monkeypatch.chdir(tmp_path)                          # an older path to a shipped prompt still works, from anywhere
    assert load("prompts/barge_in.json").ids == ("STOP", "CONTINUE")
    with pytest.raises(FileNotFoundError, match="barge_in"):
        prompt("end_of_turn")


def test_the_inputs_are_mapped_as_on_eot_bench():
    got = end_of_turn_inputs("Sure, it's nine eight four five,", "Thanks. What's the best number to reach you on?",
                             "I need an appointment")
    assert got == {"agent_question": "What's the best number to reach you on?", "caller_earlier": "I need an appointment",
                   "caller_text": "Sure, it's nine eight four five,", "transcript_end": "no sentence end"}
    assert end_of_turn_inputs("haan ji", "Okay.")["agent_question"] == "Okay.", "no question: the end of her message"
    assert end_of_turn_inputs("मैं पुणे में", punctuated=False)["transcript_end"] == UNPUNCTUATED
    assert barge_in_inputs("yes", "Dr. Rao is free on Tuesday. Shall I book it?")["agent_question"] == "Shall I book it?"
    assert dialogue_inputs("I was thinking we could", "What should we do?") == {
        "Other person last said": "What should we do?", "Speaker said": "I was thinking we could",
        "Speaker's words end with": "no sentence end"}


# -- the server ------------------------------------------------------------------------------------------------
def test_serve_downloads_the_model_starts_llama_server_and_answers(llama, monkeypatch):
    exe, model = llama
    downloads = []
    monkeypatch.setattr(server_mod, "download", lambda: downloads.append(1) or model)
    with server_mod.serve(port=free_port(), llama_server=exe) as server:
        assert downloads == [1] and server.process.poll() is None and not server.reused
        cue = Cue(server.url)
        probs = cue.end_of_turn("Sure, it's nine eight four five,", agent_said="What's the best number to reach you on?")
        assert max(probs, key=probs.get) == "MID_THOUGHT"
        assert cue.barge_in("wait, Tuesday doesn't work")["STOP"] > 0.9
        assert cue.dialogue_end_of_turn("we could go on Sunday.", other_said="What should we do?")["DONE"] > 0.9
        process = server.process
    assert process.poll() is not None, "stopped on leaving the with block"
    assert not server_mod.running(server.url)


def test_the_async_questions(running_server):
    async def main_():
        cue = Cue(running_server.url)
        probs = await cue.aend_of_turn("It's 98450 12345.", agent_said="Your number?")
        stop = await cue.abarge_in("hmm")
        dialogue = await cue.adialogue_end_of_turn("so then I", other_said="And?")
        await cue.aclose()
        return probs, stop, dialogue
    probs, stop, dialogue = asyncio.run(main_())
    assert probs["DONE"] > 0.9 and stop["CONTINUE"] > 0.9 and set(dialogue) == {"DONE", "LIKELY_DONE", "MID_THOUGHT",
                                                                               "HOLDING"}


def test_a_server_already_running_is_used_and_left_alone(running_server, llama):
    exe, model = llama
    again = Server(port=running_server.port, model=model, llama_server=exe).start()
    assert again.reused and again.process is None
    again.stop()
    assert server_mod.running(running_server.url), "not ours to stop"


def test_no_llama_server_says_how_to_install_it_before_any_download(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CUE_LLAMA_SERVER", raising=False)
    monkeypatch.setattr(server_mod, "download", lambda: pytest.fail("downloaded 8 GB with nothing to run it"))
    with pytest.raises(FileNotFoundError, match="brew install llama.cpp"):
        Server(port=free_port()).start()


def test_a_server_that_fails_to_start_shows_why(llama, tmp_path):
    exe, _ = llama
    with pytest.raises(RuntimeError, match="failed to open GGUF file"):
        Server(port=free_port(), model=tmp_path / "missing.gguf", llama_server=exe).start(timeout=20)


def test_cue_llama_server_env_points_at_the_binary(llama, monkeypatch, tmp_path):
    exe, _ = llama
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("CUE_LLAMA_SERVER", str(exe))
    assert server_mod.find_llama_server() == str(exe)


# -- the cue command -------------------------------------------------------------------------------------------
def test_cue_ask_and_demo(running_server, capsys):
    url = running_server.url
    assert main(["ask", "Sure, it's nine eight four five,", "--agent-said", "Your number?", "--url", url]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[0].split()[0] == "MID_THOUGHT" and "suggested wait" in out
    assert main(["ask", "wait, Tuesday doesn't work", "--barge-in", "--url", url]) == 0
    assert capsys.readouterr().out.strip().splitlines()[0].split()[0] == "STOP"
    assert main(["demo", "--url", url]) == 0
    out = capsys.readouterr().out
    assert out.count("\n  -> ") == 5, "three end-of-turn questions and two barge-in ones"
    assert "-> MID_THOUGHT" in out and "-> DONE" in out and "-> STOP" in out and "-> CONTINUE" in out


def test_cue_ask_with_nothing_at_an_explicit_url_says_so():
    with pytest.raises(SystemExit, match="No Cue server answers"):
        main(["ask", "hello", "--url", f"http://127.0.0.1:{free_port()}/v1"])


def start_fake(llama, port, *, load_s="0.5", model=None, api_key=None):
    """The stand-in, started directly (not through Server): a server someone else runs."""
    import subprocess
    exe, default_model = llama
    cmd = [str(exe), "-m", str(model or default_model), "--host", "127.0.0.1", "--port", str(port)]
    if api_key:
        cmd += ["--api-key", api_key]
    proc = subprocess.Popen(cmd, env={**os.environ, "FAKE_LOAD_S": load_s}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    import time as _t
    for _ in range(100):                                  # until it listens
        if server_mod.probe(f"http://127.0.0.1:{port}/v1") != "none":
            break
        _t.sleep(0.05)
    return proc


def test_a_server_still_loading_is_waited_for_not_started_again(llama, capsys):
    port = free_port()
    other = start_fake(llama, port, load_s="2.0")
    try:
        url = f"http://127.0.0.1:{port}/v1"
        assert server_mod.probe(url) == "loading"
        reused = Server(port=port, llama_server=llama[0], model=llama[1]).start(timeout=20)
        assert reused.reused and reused.process is None and server_mod.running(url)
    finally:
        other.terminate()
        other.wait()
    port = free_port()
    other = start_fake(llama, port, load_s="2.0")
    try:                                               # cue ask in the README's second terminal, while it loads
        assert main(["ask", "It's 98450 12345.", "--url", f"http://127.0.0.1:{port}/v1"]) == 0
        captured = capsys.readouterr()
        assert "still loading" in captured.err and captured.out.split()[0] == "DONE"
    finally:
        other.terminate()
        other.wait()


def test_a_port_held_by_something_else_is_refused(llama, tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class NotCue(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
    web = HTTPServer(("127.0.0.1", 0), NotCue)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RuntimeError, match="not a Cue server"):
            Server(port=web.server_address[1], model=llama[1], llama_server=llama[0]).start(timeout=20)
    finally:
        web.shutdown()
    with socket.socket() as raw:                          # something listening that is not HTTP at all
        raw.bind(("127.0.0.1", 0))
        raw.listen()
        with pytest.raises(RuntimeError, match="not a Cue server"):
            Server(port=raw.getsockname()[1], model=llama[1], llama_server=llama[0]).start(timeout=20)


def test_an_api_key_is_passed_to_llama_server_and_asked_for_when_missing(llama, capsys):
    exe, model = llama
    with Server(port=free_port(), model=model, llama_server=exe, api_key="s3cret") as server:
        assert server.command(model, str(exe))[-2:] == ["--api-key", "s3cret"]
        assert server_mod.probe(server.url) == "needs_key" and server_mod.probe(server.url, api_key="s3cret") == "ready"
        with pytest.raises(SystemExit, match="needs an API key"):
            main(["ask", "hello", "--url", server.url])
        assert main(["ask", "It's 98450 12345.", "--url", server.url, "--api-key", "s3cret"]) == 0
        assert capsys.readouterr().out.split()[0] == "DONE"


def test_a_server_that_is_not_cue_gets_a_note(llama, tmp_path, capsys):
    other_model = tmp_path / "qwen3-8b.gguf"
    other_model.write_bytes(b"GGUF")
    with Server(port=free_port(), model=other_model, llama_server=llama[0]) as server:
        main(["ask", "hello there.", "--url", server.url])
        assert "does not look like Cue v1" in capsys.readouterr().err


def test_a_named_llama_server_that_is_missing_is_an_error_not_a_swap(llama, tmp_path, monkeypatch):
    monkeypatch.setenv("CUE_LLAMA_SERVER", str(llama[0]))
    with pytest.raises(FileNotFoundError, match="llama-serverr"):
        server_mod.find_llama_server(tmp_path / "llama-serverr")
    monkeypatch.setenv("CUE_LLAMA_SERVER", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="CUE_LLAMA_SERVER"):
        server_mod.find_llama_server()


def test_ipv6_hosts_make_valid_urls():
    assert Server(host="::").url == "http://[::1]:8090/v1"
    assert Server(host="::1", port=9000).url == "http://[::1]:9000/v1"
    assert Server(host="0.0.0.0").url == "http://127.0.0.1:8090/v1"


def test_an_interrupted_start_stops_the_server_and_a_failed_download_says_so(llama, monkeypatch):
    exe, model = llama
    started = []
    real_popen = server_mod.subprocess.Popen

    def popen(*a, **k):
        started.append(real_popen(*a, **k))
        return started[-1]
    monkeypatch.setattr(server_mod.subprocess, "Popen", popen)
    real_get = server_mod.httpx.get

    def get(url, *a, **k):                                # Ctrl-C while it loads: at the first health check
        if url.endswith("/health"):
            raise KeyboardInterrupt
        return real_get(url, *a, **k)
    monkeypatch.setattr(server_mod.httpx, "get", get)
    with pytest.raises(KeyboardInterrupt):
        Server(port=free_port(), model=model, llama_server=exe).start(timeout=20)
    assert started and started[0].poll() is not None, "Ctrl-C while loading: the server is stopped, not orphaned"
    monkeypatch.undo()

    def offline():
        raise OSError("Network is unreachable")
    monkeypatch.setattr(server_mod, "download", offline)
    with pytest.raises(RuntimeError, match="could not download Pype/cue-v1"):
        Server(port=free_port(), llama_server=exe).start()


def test_python_dash_m_cue_runs_the_command(running_server):
    import subprocess
    r = subprocess.run([sys.executable, "-m", "cue", "ask", "It's 98450 12345.", "--url", running_server.url],
                       capture_output=True, text=True, timeout=30,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE.parent), os.environ.get("PYTHONPATH", "")])})
    assert r.returncode == 0 and r.stdout.split()[0] == "DONE", r.stderr
