"""Check every published eot-bench number from the saved predictions, with LiveKit's own harness and nothing of ours.

    git clone https://github.com/livekit/eot-bench && pip install -e eot-bench
    python results/eot_bench/verify.py --harness eot-bench

1. For every language and every run in results/eot_bench/runs/, the harness's `compute-metrics` is run again on the
   saved predictions.parquet, and its summary is compared with the committed metrics/summary.json.
2. Cue v1's predictions are rebuilt from its two halves (the Gemma part × √Smart Turn) and compared.
3. Cue v1's predictions are placed beside the published reference models from the harness's own output/ folder, and
   the harness's `compare-languages` ranks them. Its numbers for Cue v1 are compared with results/eot_bench/summary.json.

No GPU and no model are needed: this checks the scoring, from the predictions to every number in the README.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPAN_ROOT = "livekit__eot-bench-data__validation__min_silence_100ms"
METRICS = ["best_mean_latency_at_5pct_cutoff", "best_mean_latency_at_10pct_cutoff",
           "best_cutoff_rate_at_0_3s_latency", "best_cutoff_rate_at_0_6s_latency"]
CUE = "Shortwire Cue v1 (fine-tuned)"          # the display name the published summary.json carries


def diff(a, b, path=""):
    """Largest absolute difference between two JSON values (numbers compared, everything else must match)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return math.inf
        return max([diff(a[k], b[k], f"{path}.{k}") for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        return math.inf if len(a) != len(b) else max([diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            return 0.0 if (a != a) == (b != b) else math.inf
        return abs(a - b)
    return 0.0 if a == b else math.inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="a clone of github.com/livekit/eot-bench, pip-installed")
    a = ap.parse_args()
    harness = Path(a.harness).resolve()
    ref_root = harness / "output" / SPAN_ROOT
    tool = shutil.which("eot-harness") or str(Path(sys.executable).parent / "eot-harness")
    runs = HERE / "runs"
    tmp = Path(tempfile.mkdtemp(prefix="cue_verify_"))
    ok = True

    print("1. Recompute every run's metrics from its predictions")
    for lang_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        for run in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            out = tmp / "recompute" / lang_dir.name / run.name
            subprocess.run([tool, "compute-metrics", "--predictions", str(run / "predictions.parquet"),
                            "--output-dir", str(out)], check=True, capture_output=True)
            d = diff(json.loads((out / "summary.json").read_text()), json.loads((run / "metrics/summary.json").read_text()))
            ok &= d < 1e-9
            print(f"   {lang_dir.name}/{run.name:22} {'same' if d < 1e-9 else f'DIFFERENT (max diff {d})'}")

    print("\n2. Rebuild Cue v1 from its two halves: p(eot) = Gemma part x sqrt(Smart Turn)")
    import pandas as pd
    key = ["id", "span_index", "timestamp"]
    for lang_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        w = pd.read_parquet(lang_dir / "cue_v1_gemma_part/predictions.parquet")
        v = pd.read_parquet(lang_dir / "smart_turn_v3.2/predictions.parquet")[key + ["p_eot"]].rename(columns={"p_eot": "pv"})
        f = w.merge(v, on=key, how="left")
        f["pv"] = f.groupby(["id", "span_index"]).pv.ffill().fillna(0.5)     # Smart Turn's latest score in the pause
        f["p_eot"] = (f.p_eot * f.pv ** 0.5).clip(0, 1)
        pub = pd.read_parquet(lang_dir / "cue_v1/predictions.parquet").merge(f[key + ["p_eot"]], on=key, suffixes=("", "_rebuilt"))
        d = float((pub.p_eot - pub.p_eot_rebuilt).abs().max())
        ok &= d < 1e-9 and len(pub) == len(w)
        print(f"   {lang_dir.name}: {len(pub)} moments, {'same' if d < 1e-9 else f'DIFFERENT (max diff {d})'}")

    print("\n3. Rank Cue v1 against the published reference models")
    board = tmp / "board" / SPAN_ROOT
    for lang_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        dest = board / lang_dir.name
        dest.mkdir(parents=True)
        for item in (ref_root / lang_dir.name).iterdir():
            if item.name != "comparison":
                os.symlink(item, dest / item.name)
        os.symlink(lang_dir / "cue_v1", dest / "cue_v1")
    subprocess.run([tool, "compare-languages", str(board)], check=True, capture_output=True)
    got = json.loads((board / "language_comparison/summary.json").read_text())["values"]
    published = json.loads((HERE / "summary.json").read_text())["values"]
    worst = max(abs(got[m][l][CUE] - published[m][l][CUE]) for m in METRICS for l in published[m]
                if published[m][l].get(CUE) is not None)
    ok &= worst < 1e-9
    print(f"   Cue v1 in all 14 languages x 4 views: {'same as results/eot_bench/summary.json' if worst < 1e-9 else f'DIFFERENT (max diff {worst})'}")

    def ranked(metric, lang):
        vals = {k: v for k, v in got[metric][lang].items() if v is not None and k != "VAD baseline"}
        order = sorted(vals, key=vals.get)
        return vals, order.index(CUE) + 1, len(order)

    print("\n   English, latency at a 5% false cut-off rate:")
    vals, rank, n = ranked(METRICS[0], "en")
    for k in sorted(vals, key=vals.get):
        print(f"     {'->' if k == CUE else '  '} {k:32} {vals[k] * 1000:6.0f} ms")
    print(f"   Cue v1: {rank} of {n} in English; ", end="")
    _, rank_pt, n_pt = ranked(METRICS[0], "pt")
    print(f"{rank_pt} of {n_pt} in Portuguese")
    print(f"\n{'ALL CHECKS PASS' if ok else 'SOMETHING DIFFERS'}   (working files: {tmp})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
