"""A stand-in for llama.cpp's llama-server, for the tests: the same flags, /health (503 while "loading", then 200),
/v1/models, and one-token chat completions with logprobs. It answers MID_THOUGHT when the caller's words hold fewer
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
args, _ = ap.parse_known_args()                      # -c, -np, -ngl, --jinja: accepted as llama-server does
try:
    open(args.m).close()
except OSError:
    print(f"E gguf_init_from_file: failed to open GGUF file '{args.m}' (No such file or directory)", flush=True)
    sys.exit(1)
READY_AT = time.monotonic() + 0.5


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

    def do_GET(self):
        if self.path == "/health":
            ready = time.monotonic() >= READY_AT
            return self.reply(200 if ready else 503, {"status": "ok"} if ready else {"error": "Loading model"})
        if self.path == "/v1/models":
            return self.reply(200, {"data": [{"id": args.m}]})
        self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        letter, letters = answer(system, user)
        top = [{"token": letter, "logprob": math.log(0.97)}] + \
              [{"token": x, "logprob": math.log(0.03 / (len(letters) - 1))} for x in letters if x != letter]
        self.reply(200, {"choices": [{"message": {"content": letter}, "logprobs": {"content": [{"top_logprobs": top}]}}]})


print(f"main: server is listening on http://{args.host}:{args.port}", flush=True)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
