"""Score Cue v1 on eot-bench, beside every reference model on the same turns, with the harness's own metrics.

    git clone https://github.com/livekit/eot-bench && pip install -e eot-bench -e ".[voice]" pandas pyarrow torch torchaudio transformers
    llama-server -m cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --port 8090      # the model, from Hugging Face

    # Cue v1's words, all turns, every language (the published runs)
    python eot-results/reproduce/run.py --harness eot-bench --base-url http://localhost:8090/v1 --rows 0 \
        --lang en ar de es fr hi id it ja ko nl pt tr zh
    # Smart Turn v3.2 on the same moments, run by us (the voice half of Cue v1 x sqrt(Smart Turn))
    python eot-results/reproduce/run.py --harness eot-bench --rows 0 --lang en ... \
        --reference eot_harness.smart_turn_adapter:SmartTurnAudioAdapter
    # then combine and rank: eot-results/reproduce/leaderboard.py

The sample is a seeded random draw of whole caller turns (--rows 0: all of them), so every model is scored on
exactly the same pauses; the references' predictions come from the harness's committed output/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapter import CueEndOfTurn, ending, inputs_for, punctuation_rate  # noqa: E402

SPAN_ROOT = "livekit__eot-bench-data__validation__min_silence_100ms"


def rows_for(lang: str, n: int, seed: int, offset: int = 0, audio: bool = False) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("livekit/eot-bench-data", f"data/{lang}/validation-00000-of-00001.parquet", repo_type="dataset")
    cols = ["id", "language", "silence_spans", "messages", "words"] + (["audio"] if audio else [])
    rows = pq.read_table(path, columns=cols).to_pylist()
    random.Random(seed).shuffle(rows)
    return rows[offset:offset + n] if n else rows[offset:]


def items_for(row: dict, *, interval: float, lag: float, min_span: float, audio: bool = False):
    """The harness's causal inputs (eot_harness.io); audio is decoded only for a reference audio model."""
    from eot_harness.io import _time_grid, build_messages, decode_audio
    spans = row["silence_spans"]
    array, rate = decode_audio(row["audio"]) if audio else (None, 0)
    for i, span in enumerate(spans):
        start, end = float(span["start"]), float(span["end"])
        if end - start < min_span - 1e-9:
            continue
        label = "eot" if i == len(spans) - 1 else "hold"
        for t in _time_grid(start, end, interval):
            item = {"id": row["id"], "language": row["language"], "span_index": i, "timestamp": t,
                    "silence_dur": round(t - start, 6), "label": label,
                    "messages": build_messages(row, t, transcript_lag=lag)}
            if audio:
                item["audio"] = {"array": array[: int(t * rate + 1e-6)], "sampling_rate": rate}
            yield item


def operating_points(tradeoff: pd.DataFrame, policy: str = "model") -> dict:
    from eot_harness.metrics import get_pareto_operating_points
    sweep = tradeoff[tradeoff["policy_type"] == policy].assign(policy_type="model")
    if sweep.empty:
        return {}
    lat = get_pareto_operating_points(sweep, "cutoff_rate", [0.05, 0.10])
    cut = get_pareto_operating_points(sweep, "mean_latency", [0.3, 0.6])
    val = lambda df, b: df[df["budget_value"] == b]["value"].iloc[0]  # noqa: E731
    return {"lat@5%": val(lat, 0.05), "lat@10%": val(lat, 0.10), "cut@300ms": val(cut, 0.3), "cut@600ms": val(cut, 0.6)}


def score(df: pd.DataFrame, score_point: float | None) -> tuple[dict, pd.DataFrame]:
    from eot_harness.metrics import compute_metrics_from_predictions
    tradeoff, summary = compute_metrics_from_predictions(df, score_point_s=score_point)
    return {**operating_points(tradeoff), "auc": summary.get("auc")}, tradeoff


def references(harness: Path, lang: str, ids: set[str]) -> dict[str, dict]:
    out = {}
    for d in sorted((harness / "output" / SPAN_ROOT / lang).glob("*/predictions.parquet")):
        manifest = json.loads((d.parent / "manifest.json").read_text())
        df = pd.read_parquet(d)
        df = df[df["id"].isin(ids)]
        if df.empty:
            continue
        try:
            out[manifest.get("adapter_id") or d.parent.name], _ = score(df, manifest.get("score_point"))
        except Exception as e:
            print(f"  (skipping {d.parent.name}: {type(e).__name__}: {e})", file=sys.stderr)
    return out


def show(lang: str, table: dict[str, dict]) -> None:
    print(f"\n{lang}: latency at a 5% / 10% false-cutoff budget, false cutoffs at a 300 / 600 ms latency budget")
    print(f"  {'model':58} {'lat@5%':>8} {'lat@10%':>8} {'cut@300':>8} {'cut@600':>8}")

    def key(kv):
        v = kv[1].get("lat@5%")
        return 99 if v is None or math.isnan(v) else v

    for name, r in sorted(table.items(), key=key):
        ms = lambda v: "-" if v is None or math.isnan(v) else f"{v * 1000:.0f} ms"  # noqa: E731
        pc = lambda v: "-" if v is None or math.isnan(v) else f"{v:.1%}"         # noqa: E731
        print(f"  {name[:58]:58} {ms(r.get('lat@5%')):>8} {ms(r.get('lat@10%')):>8} "
              f"{pc(r.get('cut@300ms')):>8} {pc(r.get('cut@600ms')):>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="a checkout of github.com/livekit/eot-bench (for its output/)")
    ap.add_argument("--lang", nargs="+", default=["en"])
    ap.add_argument("--rows", type=int, default=50, help="caller turns per language; 0 = all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8090/v1", help="an OpenAI-compatible server with logprobs")
    ap.add_argument("--model", default="cue")
    ap.add_argument("--decision", default=None, help="a decision JSON (default: the shipped end_of_turn_phone prompt)")
    ap.add_argument("--empty", type=float, default=0.5, help="p(eot) before any word is visible")
    ap.add_argument("--reference", default=None,
                    help="run a harness adapter instead (module:Class, e.g. eot_harness.smart_turn_adapter:SmartTurnAudioAdapter)")
    ap.add_argument("--punctuated", default="auto", choices=["auto", "yes", "no"],
                    help="whether the recogniser punctuates the language; auto: at least 60%% of caller turns do")
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--lag", type=float, default=0.5)
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    harness = Path(a.harness).resolve()
    audio = bool(a.reference)
    if a.reference:
        from eot_harness.adapters import load_adapter
        adapter = load_adapter(a.reference)
        adapter.asked = adapter.errors = 0
    else:
        adapter = CueEndOfTurn(a.base_url, a.model, a.decision, empty=a.empty)
    print(f"{adapter.adapter_id}, {a.rows or 'all'} turns per language, seed {a.seed}, offset {a.offset}")
    for lang in a.lang:
        rows = rows_for(lang, a.rows, a.seed, a.offset, audio=audio)
        if hasattr(adapter, "punctuated"):
            rate = punctuation_rate([" ".join(w["word"] for w in r["words"]) for r in rows])
            adapter.punctuated = rate >= 0.6 if a.punctuated == "auto" else a.punctuated == "yes"
            print(f"{lang}: {rate:.0%} of caller turns end in punctuation -> "
                  f"{'punctuated' if adapter.punctuated else 'treated as unpunctuated'}")
        items = [it for r in rows for it in items_for(r, interval=a.interval, lag=a.lag, min_span=0.1, audio=audio)]
        t0 = time.monotonic()
        scores = []
        for i in range(0, len(items), 256):
            scores += adapter.predict_batch(items[i:i + 256])
        took = time.monotonic() - t0
        df = pd.DataFrame([{k: it[k] for k in ("id", "language", "span_index", "timestamp", "silence_dur", "label")}
                           for it in items])
        df["p_eot"] = scores
        if not audio:
            df["ending"] = [ending(inp["caller_text"]) if (inp := inputs_for(it["messages"])) else "nothing yet"
                            for it in items]
        sample = f"seed{a.seed}_from{a.offset}_n{a.rows or 'all'}"
        out = Path(a.out) / lang / sample / adapter.adapter_id.replace("/", "__").replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "predictions.parquet")
        spans = df.groupby(["id", "span_index"]).size().shape[0]
        print(f"\n{lang}: {len(rows)} turns, {spans} pauses, {len(items)} scored moments, "
              f"{adapter.asked} model calls ({adapter.errors} failed) in {took:.0f} s")
        ours, tradeoff = score(df, None)
        table = {f"* {adapter.adapter_id}": ours, "VAD baseline (silence only)": operating_points(tradeoff, "vad"),
                 **references(harness, lang, {r["id"] for r in rows})}
        show(lang, table)
        (out / "summary.json").write_text(json.dumps(table, indent=1, default=float))
        adapter.asked = adapter.errors = 0


if __name__ == "__main__":
    main()
