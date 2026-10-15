"""Cue v1 x sqrt(Smart Turn v3.2), built from saved runs (run.py), never re-asked, into a span-set folder laid out
like the harness's output/, with every reference model linked beside it, so the harness's own commands rank it:

    p(eot) = p_words x sqrt(p_voice)

p_words is Cue v1's end-of-turn answer (DONE + LIKELY_DONE); p_voice is Smart Turn v3.2 run by us on the same
moments. The square root makes the voice count half as much as the words (log p = log p_words + 0.5 log p_voice);
chosen once on 150 English development turns, then fixed for every turn and language.

    python eot-results/reproduce/leaderboard.py --harness eot-bench
    # -> eot-results/reproduce/out/leaderboard/<span set>/language_comparison/ (report.md, summary.json, heatmaps)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import SPAN_ROOT  # noqa: E402

KEY = ["id", "span_index", "timestamp"]
COLS = ["id", "language", "span_index", "timestamp", "silence_dur", "label"]
WORDS = "cue__end_of_turn-bb30ae78__cue+empty0.5"
VOICE = "pipecat-ai__smart-turn-v3-smart-turn-v3.2-gpu"
LANGS = "en ar de es fr hi id it ja ko nl pt tr zh".split()


def run_of(out: Path, lang: str, run: str, sample: str) -> pd.DataFrame | None:
    p = out / lang / sample / run / "predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


def combine(words: pd.DataFrame, voice: pd.DataFrame) -> pd.DataFrame:
    w = words[COLS + ["p_eot"]].rename(columns={"p_eot": "pw"})
    w = w.merge(voice[KEY + ["p_eot"]].rename(columns={"p_eot": "pv"}), on=KEY, how="left")
    w["pv"] = w.groupby(["id", "span_index"]).pv.ffill().fillna(0.5)
    w["p_eot"] = (w.pw * w.pv ** 0.5).clip(0, 1)
    return w[COLS + ["p_eot"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="a checkout of github.com/livekit/eot-bench")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--lang", nargs="+", default=LANGS)
    ap.add_argument("--sample", default="seed7_from0_nall")
    ap.add_argument("--words", default=WORDS, help="the words run's folder name")
    ap.add_argument("--voice", default=VOICE, help="the audio model run's folder name")
    a = ap.parse_args()
    harness, out = Path(a.harness).resolve(), Path(a.out)
    ref_root = harness / "output" / SPAN_ROOT
    root = out / "leaderboard" / SPAN_ROOT
    tool = Path(sys.executable).parent / "eot-harness"
    for lang in a.lang:
        words, voice = run_of(out, lang, a.words, a.sample), run_of(out, lang, a.voice, a.sample)
        if words is None or voice is None:
            print(f"{lang}: missing {'words' if words is None else 'voice'} run, skipped")
            continue
        d = root / lang
        d.mkdir(parents=True, exist_ok=True)
        for f in ["span_set.parquet", "span_set_manifest.json"]:
            if not (d / f).exists():
                os.symlink(ref_root / lang / f, d / f)
        for m in (ref_root / lang).iterdir():          # the reference models, as published
            if m.is_dir() and (m / "manifest.json").exists() and not (d / m.name).exists():
                os.symlink(m, d / m.name)
        pred = combine(words, voice)
        m = d / "cue_v1"
        m.mkdir(exist_ok=True)
        pred.to_parquet(m / "predictions.parquet", index=False)
        span_set = json.loads((ref_root / lang / "span_set_manifest.json").read_text())
        model = {"adapter_id": "pype/cue-v1", "display_name": "Cue v1", "inference_interval": 0.1,
                 "transcript_lag": 0.5, "score_point": None}
        (m / "manifest.json").write_text(json.dumps({
            "harness_version": "0.1.0", "span_set_id": span_set.get("span_set_id"), "model_run_id": "cue_v1",
            "adapter": "eot-results/reproduce/leaderboard.py", **{k: model[k] for k in ("adapter_id", "display_name")},
            "inference_interval": 0.1, "transcript_lag": 0.5, "score_point": None, "language": lang,
            "n_rows": len(pred), "n_spans": int(pred.groupby(["id", "span_index"]).ngroups),
            "dataset": {"path": "livekit/eot-bench-data", "split": "validation", "min_silence_span": 0.1, "language": lang},
            "model": model, "recipe": f"{a.words} x sqrt({a.voice})"}, indent=2))
        subprocess.run([str(tool), "compute-metrics", "--predictions", str(m / "predictions.parquet"),
                        "--output-dir", str(m / "metrics")], check=True, capture_output=True)
        print(f"{lang}: {len(pred)} moments")
    subprocess.run([str(tool), "compare-languages", str(root)], check=True)
    print(f"\n{root / 'language_comparison'}")


if __name__ == "__main__":
    main()
