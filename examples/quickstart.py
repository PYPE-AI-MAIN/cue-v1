"""Cue v1 in five questions, from Python. It uses the server at http://localhost:8090/v1 if one is running
(`cue serve`), and otherwise starts one for the run (the first time downloads the model from Hugging Face, about 8 GB).

    python examples/quickstart.py
"""
from cue import Cue, p_end, wait
from cue.server import Server, running

URL = "http://localhost:8090/v1"
server = None if running(URL) else Server().start()
cue = Cue(URL if server is None else server.url)

try:
    for question, caller in [
        ("What's the best number to reach you on?", "Sure, it's nine eight four five,"),
        ("What's the best number to reach you on?", "It's nine eight four five six seven one two three four."),
        ("How can I help you today?", "So last week I went to the clinic and the doctor told me that"),
    ]:
        probs = cue.end_of_turn(caller, agent_said=question)
        best = max(probs, key=probs.get)
        print(f"{caller!r:62} -> {best:12} p(done) {p_end(probs):.2f}, wait {wait(probs):.2f} s")

    for sound in ("wait, Tuesday doesn't work", "hmm"):
        probs = cue.barge_in(sound, agent_said="Your appointment is on Tuesday at ten")
        best = max(probs, key=probs.get)
        print(f"agent talking, caller says {sound!r:30} -> {best} ({probs[best]:.3f})")
finally:
    if server is not None:
        server.stop()
