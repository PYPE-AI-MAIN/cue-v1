"""How often the agent cuts in at the same average reply time: Cue v1, a silence timer and Smart Turn v3.2, scored with
eot-bench's own metric functions. These are the numbers in cut_ins.json and charts/cue-v1-cut-ins.png.

The leaderboard reports false cut-offs at 300 and 600 ms of latency; this reads them at any average reply time (0.8 s
is a common silence wait). For each language it takes the lowest cut-off rate whose mean latency fits the budget, for:
- Cue v1: runs/<language>/cue_v1;
- a silence timer: the silence-only policy the harness scores on the same pauses;
- Smart Turn v3.2: its published run in the harness's output/ folder.
At 0.6 s it matches the leaderboard. No GPU or model needed.

    git clone https://github.com/livekit/eot-bench && pip install -e eot-bench
    python eot-results/cut_ins.py --harness eot-bench              # recomputes and compares with cut_ins.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPAN_ROOT = "livekit__eot-bench-data__validation__min_silence_100ms"
BUDGETS = [0.6, 0.8, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="a clone of github.com/livekit/eot-bench, pip-installed")
    ap.add_argument("--write", action="store_true", help="rewrite cut_ins.json instead of comparing with it")
    a = ap.parse_args()
    import pandas as pd
    from eot_harness.metrics import compute_metrics_from_predictions, get_pareto_operating_points

    ref_root = Path(a.harness).resolve() / "output" / SPAN_ROOT

    def lowest(table, budgets):
        cut = get_pareto_operating_points(table, "mean_latency", budgets)
        return {str(b): float(cut[cut.budget_value == b]["value"].iloc[0]) for b in budgets}

    result = {"budgets_s": BUDGETS, "languages": {}}
    for lang_dir in sorted(p for p in (HERE / "runs").iterdir() if p.is_dir()):
        t, _ = compute_metrics_from_predictions(pd.read_parquet(lang_dir / "cue_v1/predictions.parquet"),
                                                score_point_s=None)
        row = {"cue_v1": lowest(t[t.policy_type == "model"], BUDGETS),
               "silence_timer": lowest(t[t.policy_type == "vad"].assign(policy_type="model"), BUDGETS)}
        st = next((d for d in sorted((ref_root / lang_dir.name).glob("smart_turn_audio_adapter__*"))
                   if (d / "predictions.parquet").exists()), None)
        if st is not None:                       # Smart Turn v3.2's published run, scored at its own score point
            sp = json.loads((st / "manifest.json").read_text()).get("score_point")
            ts, _ = compute_metrics_from_predictions(pd.read_parquet(st / "predictions.parquet"), score_point_s=sp)
            row["smart_turn_v3_2"] = lowest(ts[ts.policy_type == "model"], BUDGETS)
        result["languages"][lang_dir.name] = row

    path = HERE / "cut_ins.json"
    if a.write:
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {path}")
        return
    saved = json.loads(path.read_text())["languages"]
    got = result["languages"]

    def same(a: float, b: float) -> bool:                # NaN (no operating point within the budget) only equals NaN
        return math.isnan(a) == math.isnan(b) and (math.isnan(a) or abs(a - b) < 1e-9)

    def keys(langs: dict) -> set[tuple[str, str, str]]:
        return {(lang, who, b) for lang, row in langs.items() for who, by in row.items() for b in by}

    differ = sorted(keys(saved) ^ keys(got))               # a language, series or budget on one side only
    differ += sorted(k for k in keys(saved) & keys(got) if not same(got[k[0]][k[1]][k[2]], saved[k[0]][k[1]][k[2]]))
    for b in map(str, BUDGETS):
        mean = {who: sum(r[who][b] for r in got.values() if who in r) / max(1, sum(who in r for r in got.values()))
                for who in ("silence_timer", "smart_turn_v3_2", "cue_v1")}
        print(f"{float(b):.1f} s: silence timer {mean['silence_timer']:.1%}, Smart Turn v3.2 "
              f"{mean['smart_turn_v3_2']:.1%}, Cue v1 {mean['cue_v1']:.1%}  (average of {len(got)} languages)")
    if differ:
        print(f"DIFFERENT from cut_ins.json in {len(differ)} values, e.g. {', '.join('/'.join(k) for k in differ[:5])}")
    else:
        print("SAME AS cut_ins.json")
    raise SystemExit(1 if differ else 0)


if __name__ == "__main__":
    main()
