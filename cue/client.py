"""Ask Cue v1 a decision through any OpenAI-compatible server that returns logprobs (llama.cpp's llama-server, vLLM).

One question is one forward pass and one token: the answer is the first token's probability mass on each option's
letter, normalised over the options. Nothing is generated beyond that token.

    from cue import Cue, load
    cue = Cue("http://localhost:8090/v1")
    eot = load("prompts/end_of_turn_phone.json")
    probs = cue.ask(eot, {"agent_question": "...", "caller_earlier": "", "caller_text": "...",
                          "transcript_end": "no sentence end"})
"""
from __future__ import annotations

import math

import httpx

from .decision import LETTERS, Decision, messages

# Gemma 4 thinks before answering unless told not to; both spellings, for llama.cpp and vLLM
NO_THINKING = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def _letter(token: str) -> str | None:
    t = token.strip().strip(".:)(\"'")
    return t if len(t) == 1 and t in LETTERS else None


def body(decision: Decision, inputs: dict, model: str = "cue") -> dict:
    return {"model": model, "messages": messages(decision, inputs), "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20, **NO_THINKING}


def read(decision: Decision, response: dict) -> dict[str, float]:
    """Option probabilities from a chat-completions response."""
    ch = response["choices"][0]
    raw: dict[str, float] = {}
    content = ((ch.get("logprobs") or {}).get("content") or [])
    for tp in (content[0].get("top_logprobs") or []) if content else []:
        letter = _letter(tp["token"])
        opt = decision.option_for(letter) if letter else None
        if opt:
            raw[opt] = raw.get(opt, 0.0) + math.exp(tp["logprob"])
    if not raw:                       # no logprobs from this server: fall back to the letter it wrote
        letter = _letter((ch.get("message") or {}).get("content") or "")
        opt = decision.option_for(letter) if letter else None
        if opt is None:
            raise ValueError(f"no option letter in the answer to {decision.name}")
        raw = {opt: 1.0}
    mass = sum(raw.values())
    return {o: raw.get(o, 0.0) / mass for o in decision.ids}


class Cue:
    def __init__(self, base_url: str = "http://localhost:8090/v1", model: str = "cue", api_key: str = "x",
                 timeout: float = 30.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self._async: httpx.AsyncClient | None = None

    def ask(self, decision: Decision, inputs: dict) -> dict[str, float]:
        r = httpx.post(self.url, json=body(decision, inputs, self.model), headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return read(decision, r.json())

    async def aask(self, decision: Decision, inputs: dict) -> dict[str, float]:
        if self._async is None:
            self._async = httpx.AsyncClient(timeout=self.timeout)
        r = await self._async.post(self.url, json=body(decision, inputs, self.model), headers=self.headers)
        r.raise_for_status()
        return read(decision, r.json())

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None


# --- using the answer ------------------------------------------------------------------------------------------

REPLY = ("DONE", "LIKELY_DONE")          # the end-of-turn options that mean "the agent may speak"
WAITS = {"DONE": 0.5, "LIKELY_DONE": 1.0, "MID_THOUGHT": 1.75, "HOLDING": 2.25}   # seconds of silence


def p_end(probs: dict[str, float]) -> float:
    """p(the speaker is done): the mass on DONE and LIKELY_DONE."""
    return float(sum(probs.get(o, 0.0) for o in REPLY))


def with_voice(probs: dict[str, float], p_voice: float) -> float:
    """Cue v1's eot-bench recipe: p_end x sqrt(an audio turn model's score, e.g. Smart Turn v3.2)."""
    return p_end(probs) * max(0.0, p_voice) ** 0.5


def wait(probs: dict[str, float], waits: dict[str, float] = WAITS) -> float:
    """How long to keep listening in silence, blended by the answer's probabilities (the TurnBench policy)."""
    total = sum(probs.get(o, 0.0) for o in waits) or 1.0
    return sum(probs.get(o, 0.0) * w for o, w in waits.items()) / total
