"""A stand-in for llama.cpp's llama-server, for the tests: the same flags, /health and /v1/models (503 on both while
it "loads", as llama-server answers every endpoint until its model is ready), --api-key, and one-token chat
completions with logprobs. FAKE_LOAD_S sets the load time (default 0.5 s). It answers MID_THOUGHT when the caller's words hold fewer
than ten digits, DONE otherwise; STOP when they hold "wait", CONTINUE otherwise. A missing -m model makes it exit 1."""
import argparse
import json
import math
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("-m", required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--api-key")
args, _ = ap.parse_known_args()                      # -c, -np, -ngl, --jinja: accepted as llama-server does
try:
    open(args.m).close()
except OSError:
    print(f"E gguf_init_from_file: failed to open GGUF file '{args.m}' (No such file or directory)", flush=True)
    sys.exit(1)
import os  # noqa: E402

READY_AT = time.monotonic() + float(os.environ.get("FAKE_LOAD_S", "0.5"))


def answer(system: str, user: str) -> tuple[str, list[str]]:
    caller = re.search(r"(?:Caller|Speaker) said: (.*)", user).group(1)
    if system.startswith("The agent is speaking"):
        return ("A" if "wait" in caller.lower() else "B"), ["A", "B"]
    digits = len(re.findall(r"\d", caller)) + len(re.findall(r"\b(zero|one|two|three|four|five|six|seven|eight|nine)\b",
                                                            caller.lower()))
    return ("C" if 0 < digits < 10 else "A"), ["A", "B", "C", "D"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def gate(self):
        """503 while loading; 401 without the key (llama-server keeps /health public)."""
        if time.monotonic() < READY_AT:
            self.reply(503, {"error": "Loading model"})
            return False
        if args.api_key and self.path != "/health" and self.headers.get("Authorization") != f"Bearer {args.api_key}":
            self.reply(401, {"error": "Invalid API Key"})
            return False
        return True

    def do_GET(self):
        if not self.gate():
            return
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self.reply(200, {"data": [{"id": args.m}]})
        self.reply(404, {})

    def do_POST(self):
        if not self.gate():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        letter, letters = answer(system, user)
        top = [{"token": letter, "logprob": math.log(0.97)}] + \
              [{"token": x, "logprob": math.log(0.03 / (len(letters) - 1))} for x in letters if x != letter]
        self.reply(200, {"choices": [{"message": {"content": letter}, "logprobs": {"content": [{"top_logprobs": top}]}}]})


print(f"main: server is listening on http://{args.host}:{args.port}", flush=True)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
