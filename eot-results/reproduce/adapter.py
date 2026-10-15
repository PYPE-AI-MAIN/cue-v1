"""eot-bench adapter (https://github.com/livekit/eot-bench): Cue v1's end-of-turn decision as a scorer.

The harness asks, every 100 ms of every pause, for p(end of turn) given what a live agent would have at that
moment: the conversation so far and the caller's words a recogniser would have delivered (transcript lag 0.5 s).
This adapter puts exactly that to the end-of-turn decision and reads p(eot) as the probability of DONE and
LIKELY_DONE. Answers change only when more words arrive, so they are cached by the inputs: a pause scored 30
times asks the model once or twice. Duck-typed against the harness's batch contract (adapter_id,
predict_batch, supports_language); it imports nothing from it.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cue import UNPUNCTUATED, Cue, ending, last_question, load, p_end, prompt  # noqa: E402


def inputs_for(messages: list[dict]) -> dict[str, str] | None:
    """The decision's inputs from the harness's causal messages: the caller's words so far are the last message
    when it is the user's, the agent's question comes from the assistant message before them, and the caller's
    previous turn before that. None while no words are visible."""
    if not messages or messages[-1].get("role") != "user":
        return None
    caller = str(messages[-1].get("content") or "").strip()
    if not caller:
        return None
    question, earlier = "", ""
    for m in reversed(messages[:-1]):
        role, text = m.get("role"), str(m.get("content") or "").strip()
        if role == "assistant" and not question:
            question = last_question(text) or text[-200:]
        elif role == "user":
            earlier = text[-300:]
            break
    return {"agent_question": question, "caller_earlier": earlier, "caller_text": caller,
            "transcript_end": ending(caller)}


def punctuation_rate(turns: list[str]) -> float:
    """How often the caller's turns end in sentence punctuation: one number per language, from the words alone."""
    return sum(ending(t) != "no sentence end" for t in turns) / max(1, len(turns))


class CueEndOfTurn:
    def __init__(self, base_url: str, model: str = "cue", decision: str | Path | None = None,
                 concurrency: int = 2, empty: float = 0.5) -> None:
        self.decision = load(decision or prompt("end_of_turn_phone"))
        self.cue = Cue(base_url, model)
        self.empty = empty                    # p(eot) before any word is visible: about as often an end as not
        self.punctuated = True                # False: the recogniser seldom punctuates this language
        self.adapter_id = f"cue/{self.decision.name}-{self.decision.version}/{model}+empty{empty:g}"
        self.concurrency = concurrency
        self.cache: dict[str, dict[str, float]] = {}
        self.asked = self.errors = 0

    def supports_language(self, lang_code: str) -> bool:
        return True

    def predict_batch(self, batch: list[dict]) -> list[float]:
        keys = []
        for item in batch:
            inp = inputs_for(item.get("messages") or [])
            if inp is not None:
                if not self.punctuated and inp["transcript_end"] == "no sentence end":
                    inp["transcript_end"] = UNPUNCTUATED
                inp = {k: v for k, v in inp.items() if k in self.decision.inputs or k == "caller_text"}
            keys.append(None if inp is None else json.dumps(inp, ensure_ascii=False, sort_keys=True))
        todo = sorted({k for k in keys if k is not None and k not in self.cache})
        if todo:
            asyncio.run(self._ask_all(todo))
        return [self.empty if k is None else p_end(self.cache.get(k, {})) for k in keys]

    async def _ask_all(self, todo: list[str]) -> None:
        gate = asyncio.Semaphore(self.concurrency)

        async def one(key: str) -> None:
            async with gate:
                for attempt in range(3):
                    try:
                        self.cache[key] = await self.cue.aask(self.decision, json.loads(key))
                        self.asked += 1
                        return
                    except Exception as e:                 # a flaky request: retry, then count it as failed
                        if attempt == 2:
                            self.errors += 1
                            self.cache[key] = {}
                            print(f"  ! {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                        await asyncio.sleep(0.5 * (attempt + 1))

        await asyncio.gather(*(one(k) for k in todo))
        await self.cue.aclose()                          # each batch runs in its own event loop
