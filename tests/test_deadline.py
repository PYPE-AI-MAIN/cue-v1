"""Each prompt's deadline_ms and default are kept: a late, failed or unreadable answer gets the default (#14)."""
import asyncio
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from cue import Cue, fallback, load, parse

PHONE = load("end_of_turn_phone")       # deadline_ms 400, default DONE
BARGE = load("barge_in")                # deadline_ms 400, default CONTINUE
CALLER = "So last week I went to the clinic and the doctor told me that"


def answer(letter: str) -> dict:
    top = [{"token": letter, "logprob": math.log(0.97)}, {"token": "Z", "logprob": math.log(0.03)}]
    return {"choices": [{"message": {"content": letter}, "logprobs": {"content": [{"top_logprobs": top}]}}]}


@pytest.fixture
def server():
    """A chat-completions server whose behaviour each test sets: delay, status and body."""
    state = {"delay": 0.0, "status": 200, "body": answer("C")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(state["delay"])
            data = json.dumps(state["body"]).encode()
            try:
                self.send_response(state["status"])
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:          # the client gave up first
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    yield state
    httpd.shutdown()


def test_shipped_prompts_carry_their_deadline():
    for name in ("end_of_turn_phone", "end_of_turn_phone_greeting", "end_of_turn_dialogue", "barge_in"):
        assert load(name).deadline_ms == 400


def test_a_prompt_without_a_deadline_uses_the_timeout():
    d = parse({"name": "x", "instruction": "?", "options": [{"id": "A", "describe": "a"}, {"id": "B", "describe": "b"}]})
    assert d.deadline_ms is None
    assert Cue("http://unused/v1", timeout=7.0).deadline_s(d) == 7.0


@pytest.mark.parametrize("bad", [0, -5, "400", 1.5, True])
def test_a_bad_deadline_is_refused(bad):
    with pytest.raises(ValueError, match="deadline_ms"):
        parse({"name": "x", "instruction": "?", "deadline_ms": bad,
               "options": [{"id": "A", "describe": "a"}, {"id": "B", "describe": "b"}]})


def test_an_answer_in_time_is_used(server):
    cue = Cue(server["url"])
    probs = cue.end_of_turn(CALLER, agent_said="How can I help you today?")
    assert max(probs, key=probs.get) == "MID_THOUGHT"
    assert cue.last_fallback is None


def test_a_late_answer_gets_the_default_within_the_deadline(server):
    server["delay"] = 2.0
    cue = Cue(server["url"])
    t0 = time.perf_counter()
    probs = cue.end_of_turn(CALLER, agent_said="How can I help you today?")
    assert time.perf_counter() - t0 < 1.5                 # 400 ms, not 2 s (or the 30 s timeout)
    assert probs == fallback(PHONE) == {"DONE": 1.0, "LIKELY_DONE": 0.0, "MID_THOUGHT": 0.0, "HOLDING": 0.0}
    assert cue.last_fallback["default"] == "DONE"
    assert cue.last_fallback["reason"] in ("ReadTimeout", "TimeoutException")


def test_barge_in_falls_back_to_continue(server):
    server["delay"] = 2.0
    cue = Cue(server["url"])
    assert cue.barge_in("wait, Tuesday doesn't work", agent_said="Your appointment is on Tuesday") == fallback(BARGE)
    assert cue.last_fallback["default"] == "CONTINUE"


def test_an_error_status_gets_the_default(server):
    server["status"], server["body"] = 500, {"error": "boom"}
    cue = Cue(server["url"])
    assert cue.end_of_turn(CALLER) == fallback(PHONE)
    assert cue.last_fallback["reason"] == "HTTPStatusError"


def test_a_reply_without_an_option_letter_gets_the_default(server):
    server["body"] = {"choices": [{"message": {"content": "Hello"}, "logprobs": None}]}
    cue = Cue(server["url"])
    assert cue.end_of_turn(CALLER) == fallback(PHONE)
    assert cue.last_fallback["reason"] == "ValueError"


def test_no_server_gets_the_default():
    cue = Cue("http://127.0.0.1:9/v1")                     # nothing listens on port 9
    assert cue.end_of_turn(CALLER) == fallback(PHONE)
    assert cue.last_fallback["reason"] == "ConnectError"


def test_a_deadline_of_your_own_overrides_the_prompts(server):
    server["delay"] = 0.6
    assert Cue(server["url"]).end_of_turn(CALLER) == fallback(PHONE)               # 400 ms: too late
    assert Cue(server["url"], deadline=2.0).end_of_turn(CALLER) != fallback(PHONE)  # 2 s: in time


def test_strict_raises_and_waits_for_the_timeout(server):
    server["delay"] = 0.6
    cue = Cue(server["url"], strict=True, timeout=5.0)
    probs = cue.end_of_turn(CALLER)                                                 # past 400 ms, still answered
    assert max(probs, key=probs.get) == "MID_THOUGHT"
    server["status"], server["body"] = 500, {"error": "boom"}
    with pytest.raises(httpx.HTTPStatusError):
        cue.end_of_turn(CALLER)


def test_async_late_answer_gets_the_default(server):
    server["delay"] = 2.0

    async def run():
        cue = Cue(server["url"])
        t0 = time.perf_counter()
        probs = await cue.aend_of_turn(CALLER)
        elapsed = time.perf_counter() - t0
        await cue.aclose()
        return cue, probs, elapsed

    cue, probs, elapsed = asyncio.run(run())
    assert elapsed < 1.5
    assert probs == fallback(PHONE)
    assert cue.last_fallback["default"] == "DONE"


def test_async_answer_in_time_is_used(server):
    async def run():
        cue = Cue(server["url"])
        probs = await cue.aend_of_turn(CALLER)
        await cue.aclose()
        return cue, probs

    cue, probs = asyncio.run(run())
    assert max(probs, key=probs.get) == "MID_THOUGHT"
    assert cue.last_fallback is None
