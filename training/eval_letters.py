"""Score a decider on held-out examples: the letter it gives the most probability, and done-vs-not for end_of_turn.

    python3 eval_letters.py test_lean_v5.jsonl [--base http://localhost:8090/v1] [--model cue]

Sends each example's own messages (system + user, rendered by cue.messages) and reads the first token's letter
probabilities, as the live decider does.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math

import httpx

DONE = {"A", "B"}          # end_of_turn: A DONE, B LIKELY_DONE | C MID_THOUGHT, D HOLDING


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base", default="http://localhost:8090/v1")
    ap.add_argument("--model", default="cue")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path)]
    rows = rows[:a.limit] if a.limit else rows
    gate = asyncio.Semaphore(4)
    results = []
    async with httpx.AsyncClient(timeout=60) as http:
        async def one(r):
            body = {"model": a.model, "messages": r["messages"][:-1], "max_tokens": 1, "temperature": 0,
                    "logprobs": True, "top_logprobs": 20, "reasoning_effort": "none"}
            async with gate:
                for attempt in range(3):
                    try:
                        resp = await http.post(f"{a.base}/chat/completions", json=body)
                        resp.raise_for_status()
                        top = resp.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                        probs = collections.Counter()
                        for t in top:
                            k = t["token"].strip().strip(".:)(")
                            if len(k) == 1 and k.isalpha() and k.isupper():
                                probs[k] += math.exp(t["logprob"])
                        results.append((r, probs))
                        return
                    except Exception:
                        await asyncio.sleep(1 + attempt)
        await asyncio.gather(*(one(r) for r in rows))
    by = collections.defaultdict(lambda: [0, 0])
    binary = [0, 0]
    confusion = collections.Counter()
    for r, probs in results:
        d, gold = r["meta"]["decision"], r["messages"][-1]["content"]
        pred = max(probs, key=probs.get) if probs else "?"
        by[d][0] += pred == gold
        by[d][1] += 1
        if d.startswith("eot") or d == "end_of_turn":
            p_done = sum(probs.get(k, 0) for k in DONE) / (sum(probs.values()) or 1)
            binary[0] += (p_done >= 0.5) == (gold in DONE)
            binary[1] += 1
            confusion[(gold, pred)] += 1
    print(f"{len(results)}/{len(rows)} answered")
    for d, (ok, n) in sorted(by.items()):
        print(f"  {d:13} exact letter {ok / n:.1%}  (n={n})")
    if binary[1]:
        print(f"  end of turn, done vs not done: {binary[0] / binary[1]:.1%}")
        print("  confusion (gold -> predicted): " + ", ".join(f"{g}->{p} {c}" for (g, p), c in sorted(confusion.items())))


if __name__ == "__main__":
    asyncio.run(main())
