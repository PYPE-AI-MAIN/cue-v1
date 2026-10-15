"""A decision: one question with fixed answers, rendered into the prompt Cue v1 was trained and scored on.

The system message is the decision itself: the instruction, the options as letters, any worked examples. The user
message is this moment's inputs, in the same layout the examples use, ending where the model writes its letter.
This is the exact rendering used on live calls, on eot-bench and in training (tests/ hold the proof).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# how each input is labelled for the model; anything not listed is shown under its own name
LABELS = {
    "agent_said": "Agent is saying",
    "agent_question": "Agent asked",
    "agent_last": "Agent last said",
    "caller_text": "Caller said",
    "caller_earlier": "Caller earlier",
}


@dataclass(frozen=True)
class Option:
    id: str              # DONE, MID_THOUGHT, STOP...
    describe: str        # what it means: the only thing the model reads about it


@dataclass(frozen=True)
class Decision:
    name: str
    instruction: str
    options: tuple[Option, ...]
    inputs: tuple[str, ...]                                   # the state it reads, in the order it is shown
    examples: tuple[tuple[dict[str, str], str], ...] = ()     # (inputs, option id)
    default: str = ""
    version: str = ""                                          # a hash of what the model sees
    context: tuple[str, ...] = ()                              # inputs left out of the worked examples

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(o.id for o in self.options)

    def letter(self, option_id: str) -> str:
        return LETTERS[self.ids.index(option_id)]

    def option_for(self, letter: str) -> str | None:
        i = LETTERS.find(letter)
        return self.options[i].id if 0 <= i < len(self.options) else None


def version_of(d: dict) -> str:
    keep = {k: d.get(k) for k in ("instruction", "options", "inputs", "examples", "default", "deadline_ms")}
    keep["gate"] = None if d.get("gate") is None else repr(float(d["gate"]))
    for k in ("split", "context"):
        if d.get(k) is not None:
            keep[k] = d[k]
    return hashlib.sha1(json.dumps(keep, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def parse(d: dict) -> Decision:
    options = tuple(Option(o["id"], o["describe"]) for o in d["options"])
    ids = [o.id for o in options]
    if len(ids) < 2 or len(set(ids)) != len(ids) or len(ids) > len(LETTERS):
        raise ValueError(f"{d.get('name')}: needs 2-26 distinct options")
    default = d.get("default") or ids[-1]
    if default not in ids:
        raise ValueError(f"{d['name']}: default {default!r} is not an option")
    examples = []
    for ex in d.get("examples", []):
        if ex["answer"] not in ids:
            raise ValueError(f"{d['name']}: example answer {ex['answer']!r} is not an option")
        examples.append(({k: str(v) for k, v in ex["inputs"].items()}, ex["answer"]))
    return Decision(name=d["name"], instruction=d["instruction"].strip(), options=options,
                    inputs=tuple(d.get("inputs", ())), examples=tuple(examples), default=default,
                    version=version_of(d), context=tuple(d.get("context", ())))


# the three prompts Cue v1 was trained on, shipped inside the package
PROMPTS = Path(__file__).resolve().parent / "prompts"


def prompt(name: str) -> Path:
    """The file of a prompt shipped with Cue v1: "end_of_turn_phone", "end_of_turn_dialogue" or "barge_in"."""
    path = PROMPTS / f"{name}.json"
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in PROMPTS.glob("*.json")))
        raise FileNotFoundError(f"no prompt {name!r}: Cue v1 ships {names}")
    return path


def load(path_or_name: str | Path) -> Decision:
    """A decision: a shipped prompt by name ("end_of_turn_phone") or a JSON file of your own. An older path to a
    shipped prompt ("prompts/end_of_turn_phone.json") still finds it."""
    path = Path(path_or_name)
    if not path.exists() and (PROMPTS / f"{path.stem}.json").exists() and path.suffix in ("", ".json"):
        path = PROMPTS / f"{path.stem}.json"
    return parse(json.loads(path.read_text()))


def inputs_block(decision: Decision, inputs: dict[str, object], *, example: bool = False) -> str:
    lines = []
    for name in decision.inputs:
        if example and name in decision.context:
            continue
        v = str(inputs.get(name, "") or "").strip()
        lines.append(f"{LABELS.get(name, name)}: {v or '(nothing)'}")
    return "\n".join(lines)


def system(decision: Decision) -> str:
    parts = [decision.instruction, "", "Options:"]
    parts += [f"{LETTERS[i]}: {o.describe}" for i, o in enumerate(decision.options)]
    parts += ["", "Answer with exactly one letter."]
    if decision.examples:
        parts += ["", "Examples:"]
        for ex_inputs, answer in decision.examples:
            parts += ["", inputs_block(decision, ex_inputs, example=True), f"Answer: {decision.letter(answer)}"]
    return "\n".join(parts)


def user(decision: Decision, inputs: dict[str, object]) -> str:
    return inputs_block(decision, inputs) + "\nAnswer:"


def messages(decision: Decision, inputs: dict[str, object]) -> list[dict[str, str]]:
    """The chat messages for one question: send these, read the first token's option letters."""
    return [{"role": "system", "content": system(decision)}, {"role": "user", "content": user(decision, inputs)}]


# --- helpers for building inputs -------------------------------------------------------------------------------

def ending(text: str) -> str:
    """How a transcript ends, read from the recogniser's punctuation, for the phone prompt's transcript_end input."""
    t = text.strip().rstrip("\"')]」』")
    if t.endswith(("?", "？", "؟")):
        return "a question mark"
    if t.endswith((".", "!", "।", "。", "！", "۔")):
        return "a full stop"
    return "no sentence end"


# for a recogniser that seldom punctuates a language: a missing full stop then says nothing
UNPUNCTUATED = "no punctuation (this recogniser rarely punctuates)"

QUESTION = re.compile(r"\?|\b(am i|are you|is this|is that|do you|did you|have you|can you|could you|shall i|"
                      r"should i|would you|will you|kya|kaun|kaunsa|kahan|kaise|kitna|kitne|kab|theek hai|right|"
                      r"sahi hai|haan ya na|okay)\b"
                      r"|(क्या|कौन|कौनसा|कहाँ|कैसे|कितना|कितने|कब|ठीक है|सही है|है ना|बताइए|बताओ)", re.I)


def last_question(agent_said: str) -> str:
    """The agent's last sentence, if it reads as a question; else ''."""
    if not agent_said:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", agent_said.strip()) if p.strip()]
    last = parts[-1] if parts else agent_said.strip()
    return last if QUESTION.search(last) else ""
