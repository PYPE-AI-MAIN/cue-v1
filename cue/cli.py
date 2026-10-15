"""The ``cue`` command: run Cue v1 and try it.

    cue serve      download Cue v1 from Hugging Face (once, about 8 GB) and serve it at http://localhost:8090/v1
    cue demo       five example decisions
    cue ask "Sure, it's nine eight four five," --agent-said "What's the best number to reach you on?"
    cue ask "wait, Tuesday doesn't work" --barge-in --agent-said "Your appointment is on Tuesday at ten"

``demo`` and ``ask`` use the server at --url, or start one for the run if none is running there.
"""
from __future__ import annotations

import argparse
import os
import sys

import httpx

from .client import Cue, p_end, wait
from .server import Server, running

DEFAULT_URL = "http://localhost:8090/v1"

DEMO_END_OF_TURN = [
    ("What's the best number to reach you on?", "Sure, it's nine eight four five,"),
    ("What's the best number to reach you on?", "It's nine eight four five six seven one two three four."),
    ("How can I help you today?", "So last week I went to the clinic and the doctor told me that"),
]
DEMO_BARGE_IN = ("Your appointment is on Tuesday at ten", ["wait, Tuesday doesn't work", "hmm"])


def _connect(url: str | None, model: str = "cue", api_key: str | None = None) -> tuple[Cue, Server | None]:
    """A client for the server at url; with no url and nothing at the default, a local server for this run."""
    if running(url or DEFAULT_URL, api_key=api_key):
        return Cue(url or DEFAULT_URL, model, api_key or "x"), None
    if url:
        raise SystemExit(f"No Cue server answers at {url}.")
    print(f"No Cue server at {DEFAULT_URL}: starting one for this run (the first time downloads the model, about "
          "8 GB). To ask many questions, run `cue serve` in another terminal.", file=sys.stderr)
    server = Server().start()
    return Cue(server.url), server


def _show(probs: dict[str, float], *, end_of_turn: bool) -> None:
    for option, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {option:12} {p:.3f}")
    if end_of_turn:
        print(f"  p(done) {p_end(probs):.2f}, suggested wait {wait(probs):.2f} s")


def serve_cmd(a: argparse.Namespace) -> int:
    server = Server(port=a.port, host=a.host, model=a.model, parallel=a.parallel, ctx=a.ctx,
                    gpu_layers=a.gpu_layers, llama_server=a.llama_server)
    print("Starting Cue v1 (the first time downloads the model from Hugging Face, about 8 GB)...", file=sys.stderr)
    server.start()
    if server.reused:
        print(f"A server already answers at {server.url}.")
        return 0
    print(f"Cue v1 is serving at {server.url}  (log: {server.log_path}). Ctrl-C to stop.")
    try:
        return server.wait()
    except KeyboardInterrupt:
        server.stop()
        return 0


def ask_cmd(a: argparse.Namespace) -> int:
    cue, server = _connect(a.url, a.model, a.api_key)
    try:
        if a.barge_in:
            probs = cue.barge_in(a.text, agent_said=a.agent_said)
        elif a.dialogue:
            probs = cue.dialogue_end_of_turn(a.text, other_said=a.agent_said)
        else:
            probs = cue.end_of_turn(a.text, agent_said=a.agent_said, caller_earlier=a.earlier,
                                    punctuated=not a.unpunctuated)
        _show(probs, end_of_turn=not a.barge_in)
    finally:
        if server is not None:
            server.stop()
    return 0


def demo_cmd(a: argparse.Namespace) -> int:
    cue, server = _connect(a.url, a.model, a.api_key)
    try:
        for question, caller in DEMO_END_OF_TURN:
            probs = cue.end_of_turn(caller, agent_said=question)
            best = max(probs, key=probs.get)
            print(f"agent: {question!r}\n  caller: {caller!r}\n  -> {best}, p(done) {p_end(probs):.2f}, "
                  f"wait {wait(probs):.2f} s\n")
        agent, sounds = DEMO_BARGE_IN
        for sound in sounds:
            probs = cue.barge_in(sound, agent_said=agent)
            best = max(probs, key=probs.get)
            print(f"agent talking: {agent!r}\n  caller over her: {sound!r}\n  -> {best} ({probs[best]:.3f})\n")
    finally:
        if server is not None:
            server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cue", description="Cue v1: turn detection for voice agents.")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="download Cue v1 (once) and serve it with llama.cpp's llama-server")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines")
    s.add_argument("--model", help="a local cue-v1-q8_0.gguf instead of the Hugging Face download")
    s.add_argument("--parallel", type=int, default=4, help="questions answered at once (llama-server -np)")
    s.add_argument("--ctx", type=int, default=8192, help="context tokens, shared by the parallel slots")
    s.add_argument("--gpu-layers", type=int, default=999, help="layers on the GPU; 0 for CPU only")
    s.add_argument("--llama-server", help="the llama-server binary, if not on PATH")
    s.set_defaults(fn=serve_cmd)

    q = sub.add_parser("ask", help="ask Cue v1 one question")
    q.add_argument("text", help="the caller's words")
    q.add_argument("--agent-said", default="", help="end of turn: the agent's last message; "
                   "--barge-in: what she has said so far; --dialogue: the other person's last words")
    q.add_argument("--earlier", default="", help="end of turn: the caller's previous message")
    q.add_argument("--barge-in", action="store_true", help="the caller spoke over the agent: STOP or CONTINUE?")
    q.add_argument("--dialogue", action="store_true", help="two people talking, not a caller and an agent")
    q.add_argument("--unpunctuated", action="store_true", help="your speech recogniser seldom punctuates")
    q.set_defaults(fn=ask_cmd)

    d = sub.add_parser("demo", help="five example decisions")
    d.set_defaults(fn=demo_cmd)

    for p in (q, d):                                  # which server to ask
        p.add_argument("--url", help=f"a running Cue server (default {DEFAULT_URL}, started if nothing answers)")
        p.add_argument("--api-key", default=os.environ.get("CUE_API_KEY"), help="if the server needs one "
                       "(default: $CUE_API_KEY)")
        p.add_argument("--model", default="cue", help="the model name the server expects (llama-server ignores it)")

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (FileNotFoundError, RuntimeError, TimeoutError) as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"The Cue server did not answer: {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
