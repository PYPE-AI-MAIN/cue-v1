"""Cue v1 in four questions. Serve the model first:

    huggingface-cli download Pype/cue-v1 cue-v1-q8_0.gguf --local-dir .
    llama-server -m cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --port 8090

    python examples/quickstart.py [http://localhost:8090/v1]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cue import Cue, ending, load, p_end, wait  # noqa: E402

cue = Cue(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8090/v1")
phone = load(ROOT / "prompts" / "end_of_turn_phone.json")
barge_in = load(ROOT / "prompts" / "barge_in.json")


def end_of_turn(question: str, caller: str) -> None:
    probs = cue.ask(phone, {"agent_question": question, "caller_earlier": "", "caller_text": caller,
                            "transcript_end": ending(caller)})
    best = max(probs, key=probs.get)
    print(f"{caller!r:52} -> {best:12} p(done) {p_end(probs):.2f}, wait {wait(probs):.2f} s")


end_of_turn("What's the best number to reach you on?", "Sure, it's nine eight four five,")
end_of_turn("What's the best number to reach you on?", "It's nine eight four five six seven one two three four.")
end_of_turn("How can I help you today?", "So last week I went to the clinic and the doctor told me that")

for sound in ("wait, Tuesday doesn't work", "hmm"):
    probs = cue.ask(barge_in, {"agent_said": "Your appointment is on Tuesday at ten", "agent_question": "",
                               "caller_text": sound})
    print(f"agent talking, caller says {sound!r:30} -> {max(probs, key=probs.get)} ({max(probs.values()):.3f})")
