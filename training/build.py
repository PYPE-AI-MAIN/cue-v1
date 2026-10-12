"""Simulated conversations (SPEC.md) -> chat-format training examples, rendered by cue.messages exactly as Cue is asked.

    python training/build.py --lean --v5 conversations/*.jsonl     # Cue v1's recipe: train_lean_v5.jsonl, test_lean_v5.jsonl

Every pause becomes an end-of-turn question, asked in more than one wording so the model is not tied to one prompt:
the two-person prompt (prompts/end_of_turn_dialogue.json), the phone prompt (prompts/end_of_turn_phone.json, with
--v5) and Shortwire's production end-of-turn prompt (training/prompts/end_of_turn_production.json). Every listener
sound becomes a barge-in question (prompts/barge_in.json). The answer is the option's letter, exactly as the model
is read from its first token. 10% of the conversations (seed 7) are held out whole as the test set.

--lean drops the worked examples from every prompt (shorter prompts, about 3x faster training on a T4) and asks phone
calls only the phone-call prompts. --v5 adds the phone prompt on the caller's pauses. Cue v1 used both.
"""
from __future__ import annotations

import dataclasses
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cue import last_question, load, messages  # noqa: E402

EOT = {"MID_THOUGHT": "MID_THOUGHT", "HOLDING": "HOLDING", "END_DONE": "DONE", "END_LIKELY": "LIKELY_DONE"}


def ending(text: str) -> str:
    t = text.strip().rstrip("\"')]")
    if t.endswith("?"):
        return "a question mark"
    if t.endswith((".", "!")):
        return "a full stop"
    return "no sentence end"


def texture(text: str, rng: random.Random) -> str:
    """A little recogniser noise on top of what was written: sometimes no final punctuation, sometimes lowercase."""
    if rng.random() < 0.12:
        text = re.sub(r"[.!]\s*$", "", text.strip())
    if rng.random() < 0.15:
        text = text[:1].lower() + text[1:]
    return text


def example(decision, inputs: dict, answer: str, meta: dict) -> dict:
    return {"messages": messages(decision, inputs) + [{"role": "assistant", "content": decision.letter(answer)}],
            "meta": {**meta, "inputs": inputs, "answer": answer}}


def build(conv: dict, decisions: dict, rng: random.Random) -> list[dict]:
    out = []
    turns = conv["turns"]
    phone = conv["type"] == "phone_agent"
    lean = decisions.get("_lean")
    said = [" ".join(c["text"].strip() for c in t["chunks"]) for t in turns]
    for i, t in enumerate(turns):
        other = said[i - 1][-240:] if i else ""
        earlier = said[i - 2][-300:] if i >= 2 else ""
        so_far = []
        for j, c in enumerate(t["chunks"]):
            so_far.append(texture(c["text"].strip(), rng))
            spoken = " ".join(so_far)[-400:]
            meta = {"type": conv["type"], "topic": conv["topic"], "turn": i, "chunk": j}
            if c["pause"] in EOT and not (conv.get("_bargein") and t["speaker"] == "A"):   # barge-in batches: caller pauses only
                label = EOT[c["pause"]]
                if not (lean and phone):
                    out.append(example(decisions["dialogue"], {"Other person last said": other, "Speaker said": spoken,
                                                               "Speaker's words end with": ending(spoken)}, label,
                                       {**meta, "decision": "eot_dialogue"}))
                question = last_question(other) or other[-200:]
                v5_here = (phone and t["speaker"] == "B") if lean else True
                if decisions["v5"] is not None and v5_here:
                    out.append(example(decisions["v5"], {"agent_question": question, "caller_earlier": earlier,
                                                         "caller_text": spoken, "transcript_end": ending(spoken)}, label,
                                       {**meta, "decision": "eot_v5"}))
                if phone and t["speaker"] == "B":
                    out.append(example(decisions["end_of_turn"], {"agent_question": question, "caller_earlier": earlier,
                                                                  "caller_text": spoken}, label,
                                       {**meta, "decision": "end_of_turn"}))
            for lst in t.get("listener", []):
                if int(lst["after_chunk"]) == j:
                    agent_said = " ".join(so_far)[-300:]
                    out.append(example(decisions["barge_in"], {"agent_said": agent_said,
                                                               "agent_question": last_question(agent_said),
                                                               "caller_text": lst["text"].strip()},
                                       "STOP" if lst["kind"] == "INTERRUPT" else "CONTINUE",
                                       {**meta, "decision": "barge_in"}))
    return out


def main(paths: list[str]) -> None:
    lean = "--lean" in paths
    v5 = "--v5" in paths
    paths = [p for p in paths if p not in ("--lean", "--v5")]
    decisions = {"dialogue": load(ROOT / "prompts" / "end_of_turn_dialogue.json"),
                 "v5": load(ROOT / "prompts" / "end_of_turn_phone.json"),
                 "end_of_turn": load(ROOT / "training" / "prompts" / "end_of_turn_production.json"),
                 "barge_in": load(ROOT / "prompts" / "barge_in.json")}
    if lean:
        decisions = {k: dataclasses.replace(d, examples=()) for k, d in decisions.items()}
        if not v5:
            decisions["v5"] = None
        decisions["_lean"] = True
    convs = [{**json.loads(line), "_bargein": Path(p).name.startswith("bargein")}
             for p in paths for line in open(p) if line.strip()]
    rng = random.Random(7)
    rng.shuffle(convs)
    n_test = max(1, len(convs) // 10)
    suffix = ("_lean" if lean else "") + ("_v5" if lean and v5 else "")
    for name, part in (("test", convs[:n_test]), ("train", convs[n_test:])):
        rows = [ex for c in part for ex in build(c, decisions, rng)]
        rng.shuffle(rows)
        with open(f"{name}{suffix}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by: dict = {}
        for r in rows:
            k = (r["meta"]["decision"], r["messages"][-1]["content"])
            by[k] = by.get(k, 0) + 1
        print(f"{name}: {len(part)} conversations -> {len(rows)} examples")
        for k in sorted(by):
            print(f"   {k[0]:13} {k[1]}: {by[k]}")


if __name__ == "__main__":
    main(sys.argv[1:])
