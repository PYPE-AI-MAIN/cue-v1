"""The prompts are rendered byte for byte as on the benchmarks and in training, and answers are read correctly.

tests/fixtures/rendered.json was written by Shortwire's own renderer; the version hashes are the ones in every
benchmark run's name (bb30ae78 on eot-bench, ac05ea9a on the two-person benchmark, d498d7e0 in training).
"""
import json
import math
from pathlib import Path

import pytest

from cue import SmartTurn, ending, load, messages, p_end, prepare_audio_window, read, system, wait, with_voice

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
    d = load(ROOT / "prompts" / "end_of_turn_phone.json")
    probs = read(d, response({"C": 0.6, " D": 0.2, "A": 0.1, "Hello": 0.05}))
    assert probs["MID_THOUGHT"] == pytest.approx(0.6 / 0.9)
    assert probs["HOLDING"] == pytest.approx(0.2 / 0.9)
    assert probs["LIKELY_DONE"] == 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_read_falls_back_to_the_written_letter():
    d = load(ROOT / "prompts" / "barge_in.json")
    r = {"choices": [{"message": {"content": "A"}, "logprobs": None}]}
    assert read(d, r) == {"STOP": 1.0, "CONTINUE": 0.0}


def test_using_the_answer():
    probs = {"DONE": 0.5, "LIKELY_DONE": 0.25, "MID_THOUGHT": 0.25, "HOLDING": 0.0}
    assert p_end(probs) == 0.75
    assert with_voice(probs, 0.25) == pytest.approx(0.375)
    assert wait(probs) == pytest.approx(0.5 * 0.5 + 0.25 * 1.0 + 0.25 * 1.75)


def test_direct_token_rendering_and_reading():
    d = load(ROOT / "prompts" / "end_of_turn_phone.json")
    sys_prompt = system(d, direct=True)
    assert "DONE: finished:" in sys_prompt
    assert "Answer: MID_THOUGHT" in sys_prompt

    # BPE tokenizers often split multi-word option IDs on the first token (e.g. "MID" for MID_THOUGHT)
    probs = read(d, response({"MID": 0.7, "DONE": 0.2, "HOLD": 0.1}, content="MID_THOUGHT"), direct=True)
    assert probs["MID_THOUGHT"] == pytest.approx(0.7)
    assert probs["DONE"] == pytest.approx(0.2)
    assert probs["HOLDING"] == pytest.approx(0.1)


def test_coding_prompt_loads_and_renders():
    d = load(ROOT / "prompts" / "end_of_turn_coding.json")
    assert d.ids == ("DONE", "LIKELY_DONE", "MID_THOUGHT", "HOLDING")
    msgs = messages(d, {"agent_question": "Which host?", "caller_earlier": "",
                        "caller_text": "Connect to 192.168.1", "transcript_end": ending("Connect to 192.168.1")})
    assert len(msgs) == 2
    assert "192.168.1" in msgs[1]["content"]


def test_smart_turn_audio_window_alignment():
    np = pytest.importorskip("numpy")
    # 1 second of 16kHz tone -> should be left-padded so the signal sits at the right edge of the 128,000-sample window
    tone = np.ones(16000, dtype=np.float32) * 2.0
    aligned = prepare_audio_window(tone, sample_rate=16000)
    assert aligned.shape == (128000,)
    assert float(np.mean(aligned)) == pytest.approx(0.0, abs=1e-5)
    assert float(np.std(aligned)) == pytest.approx(1.0, abs=1e-4)
    # Right edge (last 16k samples) has higher amplitude than left-padded region
    assert float(np.mean(aligned[-16000:])) > float(np.mean(aligned[:16000]))

    st = SmartTurn(model_path= ROOT / "nonexistent.onnx")
    assert not st.is_available
    assert st.predict(tone) == 0.5
