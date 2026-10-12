"""Check a batch of synthetic conversations against SPEC.md and print its proportions.

    python3 validate.py batch_01.jsonl [more.jsonl ...]
Exits non-zero, naming the line, on the first conversation that breaks the shape.
"""
from __future__ import annotations

import collections
import json
import sys

TYPES = {"casual", "argumentative", "collaborative", "instructional", "narrative", "transactional", "phone_agent"}
MID = {"MID_THOUGHT", "HOLDING"}
END = {"END_DONE", "END_LIKELY"}


def check(conv: dict) -> list[str]:
    errs = []
    if set(conv) - {"type", "topic", "turns"}:
        errs.append(f"unknown keys {set(conv) - {'type', 'topic', 'turns'}}")
    if conv.get("type") not in TYPES:
        errs.append(f"type {conv.get('type')!r}")
    turns = conv.get("turns") or []
    if not 10 <= len(turns) <= 32:
        errs.append(f"{len(turns)} turns")
    for i, t in enumerate(turns):
        if t.get("speaker") != ("A" if i % 2 == 0 else "B"):
            errs.append(f"turn {i}: speaker {t.get('speaker')!r}, expected alternating A/B")
        ch = t.get("chunks") or []
        if not ch or any(not str(c.get("text", "")).strip() for c in ch):
            errs.append(f"turn {i}: empty chunk")
            continue
        interrupted = any(l.get("kind") == "INTERRUPT" for l in t.get("listener", []))
        for j, c in enumerate(ch):
            last = j == len(ch) - 1
            ok = ({"INTERRUPTED"} if interrupted else END) if last else MID
            if c.get("pause") not in ok:
                errs.append(f"turn {i} chunk {j}: pause {c.get('pause')!r}, expected one of {sorted(ok)}")
            if "..." in c["text"]:
                errs.append(f"turn {i} chunk {j}: contains '...'")
        for l in t.get("listener", []):
            if l.get("kind") not in {"BACKCHANNEL", "INTERRUPT"} or not str(l.get("text", "")).strip():
                errs.append(f"turn {i}: bad listener {l}")
            elif not 0 <= int(l.get("after_chunk", -1)) < len(ch):
                errs.append(f"turn {i}: listener after_chunk {l.get('after_chunk')} out of range")
            elif l["kind"] == "INTERRUPT" and int(l["after_chunk"]) != len(ch) - 1:
                errs.append(f"turn {i}: INTERRUPT must follow the last chunk")
    return errs


def main(paths: list[str]) -> None:
    pauses, ends, kinds, types, topics = collections.Counter(), collections.Counter(), collections.Counter(), \
        collections.Counter(), []
    n = 0
    for path in paths:
        for ln, line in enumerate(open(path), 1):
            if not line.strip():
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{ln}: not JSON ({e})")
            errs = check(conv)
            if errs:
                sys.exit(f"{path}:{ln}: " + "; ".join(errs[:5]))
            n += 1
            types[conv["type"]] += 1
            topics.append(conv["topic"].strip().lower())
            for t in conv["turns"]:
                for c in t["chunks"]:
                    pauses[c["pause"]] += 1
                    if c["pause"] in END:
                        txt = c["text"].strip()
                        ends["question" if txt.endswith("?") else "statement" if txt.endswith((".", "!")) else "none"] += 1
                for l in t.get("listener", []):
                    kinds[l["kind"]] += 1
    total = sum(pauses.values())
    holds = pauses["MID_THOUGHT"] + pauses["HOLDING"]
    print(f"{n} conversations OK, {total} pauses: holds {holds / total:.0%} "
          f"(HOLDING {pauses['HOLDING'] / max(holds, 1):.0%} of holds), ends {dict(ends)}, interrupted {pauses['INTERRUPTED']}")
    print(f"listener: {dict(kinds)}   types: {dict(types)}")
    dup = [t for t, c in collections.Counter(topics).items() if c > 1]
    if dup:
        print(f"repeated topics: {dup[:5]}")


if __name__ == "__main__":
    main(sys.argv[1:])
