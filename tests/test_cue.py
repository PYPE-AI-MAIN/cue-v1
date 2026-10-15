"""The prompts are rendered byte for byte as on the benchmarks and in training, and answers are read correctly.

tests/fixtures/rendered.json was written by Shortwire's own renderer; the version hashes are the ones in every
benchmark run's name (bb30ae78 on eot-bench, ac05ea9a on the two-person benchmark).
"""
import json
import math
from pathlib import Path

import pytest

from cue import load, messages, p_end, read, wait, with_voice

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "tests" / "fixtures" / "rendered.json").read_text())


@pytest.mark.parametrize("case", FIXTURES, ids=lambda c: f"{Path(c['decision']).stem}:{c['inputs'].get('caller_text', '')[:20]}")
def test_rendering_matches_shortwire(case):
    d = load(ROOT / case["decision"])
    assert d.version == case["version"]
    assert messages(d, case["inputs"]) == case["messages"]


def response(top: dict[str, float], content: str = "A") -> dict:
    return {"choices": [{"message": {"content": content},
                         "logprobs": {"content": [{"top_logprobs": [{"token": t, "logprob": math.log(p)}
                                                                    for t, p in top.items()]}]}}]}


def test_read_normalises_over_option_letters():
    d = load("end_of_turn_phone")
    probs = read(d, response({"C": 0.6, " D": 0.2, "A": 0.1, "Hello": 0.05}))
    assert probs["MID_THOUGHT"] == pytest.approx(0.6 / 0.9)
    assert probs["HOLDING"] == pytest.approx(0.2 / 0.9)
    assert probs["LIKELY_DONE"] == 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_read_falls_back_to_the_written_letter():
    d = load("barge_in")
    r = {"choices": [{"message": {"content": "A"}, "logprobs": None}]}
    assert read(d, r) == {"STOP": 1.0, "CONTINUE": 0.0}


def test_using_the_answer():
    probs = {"DONE": 0.5, "LIKELY_DONE": 0.25, "MID_THOUGHT": 0.25, "HOLDING": 0.0}
    assert p_end(probs) == 0.75
    assert with_voice(probs, 0.25) == pytest.approx(0.375)
    assert wait(probs) == pytest.approx(0.5 * 0.5 + 0.25 * 1.0 + 0.25 * 1.75)
