"""Cue v1 launch infographics on an off-white surface: the English table and the 14-language heatmaps."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

SUMMARY = Path(sys.argv[1])
OUT = Path(sys.argv[2])
s = json.loads(SUMMARY.read_text())
V = s["values"]
L5, L10 = "best_mean_latency_at_5pct_cutoff", "best_mean_latency_at_10pct_cutoff"
C3, C6 = "best_cutoff_rate_at_0_3s_latency", "best_cutoff_rate_at_0_6s_latency"
CUE = "Shortwire Cue v1 (fine-tuned)"

SURFACE = "#faf8f3"
INK, INK2, MUTED = "#1b1b1a", "#5f5e5a", "#9a9890"
RULE = "#e6e3da"
BLUE, BLUE_TINT = "#2a78d6", "#e6f0fb"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial"], "text.color": INK,
                     "axes.edgecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

NAMES = {CUE: "Cue v1", "LiveKit Turn Detector v1": "LiveKit Turn Detector v1",
         "LiveKit Turn Detector v1-mini": "LiveKit v1-mini (local)", "SmartTurn v3.2": "Smart Turn v3.2",
         "OpenAI GPT Realtime 2": "OpenAI GPT Realtime 2", "JoinIn AI Baton": "JoinIn AI Baton",
         "VAP (silent agent)": "VAP", "VAD baseline": "Silence only (VAD)"}
LANG = {"en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
        "nl": "Dutch", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "id": "Indonesian", "hi": "Hindi",
        "tr": "Turkish", "ar": "Arabic"}


def english_table(path: Path) -> None:
    rows = []
    for m in s["models"]:
        v = [V[k]["en"].get(m) for k in (L5, L10, C3, C6)]
        if v[0] is None:
            continue
        rows.append((NAMES.get(m, m), v[0] * 1000, None if v[1] is None else v[1] * 1000,
                     None if v[2] is None else v[2] * 100, None if v[3] is None else v[3] * 100, m == CUE, False))
    rows.append(("turn-1-mini †", 849.0, None, 22.6, None, False, True))
    # Cue v1 on its own (words and context, no audio model): the same run before the √Smart Turn fusion, scored with
    # the harness's metrics (750 / 552 ms, 34.0% / 7.8%)
    rows.append(("Cue v1, Gemma part only", 749.9, 551.8, 34.0, 7.8, False, False))
    base = [r for r in rows if r[0] == "Silence only (VAD)"]
    rows = sorted([r for r in rows if r[0] != "Silence only (VAD)"], key=lambda r: r[1]) + base

    n = len(rows)
    fig = plt.figure(figsize=(12, 1.5 + 0.46 * n), dpi=200)
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.79])
    ax.set_xlim(0, 100)
    ax.set_ylim(n, -0.75)
    ax.axis("off")
    fig.text(0.04, 0.94, "Cue v1 on eot-bench, English", fontsize=20, fontweight="bold")
    fig.text(0.04, 0.905, "400 caller turns, scored with the benchmark's own harness. Lower is better in every column.",
             fontsize=11, color=INK2)
    cols = [(3, "#", "left"), (7, "Model", "left"), (38, "Latency @ 5% cut-offs", "left"), (66, "@ 10%", "right"),
            (84, "Cut-offs @ 300 ms", "right"), (97, "@ 600 ms", "right")]
    for x, label, ha in cols:
        ax.text(x, -0.45, label, fontsize=10, color=INK2, ha=ha, va="center", fontweight="bold")
    ax.plot([0, 100], [-0.05, -0.05], color=INK, lw=0.8)
    vmax = max(r[1] for r in rows)
    for i, (name, l5, l10, c3, c6, cue, reported) in enumerate(rows):
        y = i + 0.5
        if cue:
            ax.add_patch(FancyBboxPatch((0.4, i + 0.08), 99.2, 0.84, boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=BLUE_TINT, ec="none", zorder=0))
        else:
            ax.plot([0, 100], [i + 1, i + 1], color=RULE, lw=0.7)
        weight = "bold" if cue else "normal"
        ink = MUTED if name.startswith("Silence") else INK
        unranked = name.startswith("Silence") or name == "Cue v1, Gemma part only"     # a baseline, and a variant of Cue
        rank = "" if unranked else str(1 + sum(1 for r in rows[:i] if not (r[0].startswith("Silence") or r[0] == "Cue v1, Gemma part only")))
        ax.text(3, y, rank, fontsize=11, color=INK2, va="center")
        ax.text(7, y, name, fontsize=12, color=ink, va="center", fontweight=weight, style="italic" if reported else "normal")
        w = 14 * l5 / vmax                                  # inline bar for the headline column
        ax.add_patch(FancyBboxPatch((38, y - 0.13), w, 0.26, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=BLUE if cue else ("#9ec5f4" if name == "Cue v1, Gemma part only" else ("#c9c6bc" if not name.startswith("Silence") else "#e2dfd6")),
                                    ec="none"))
        ax.text(38 + w + 0.8, y, f"{l5:,.0f} ms", fontsize=11, va="center", color=ink, fontweight=weight)
        for x, val, unit in ((66, l10, " ms"), (84, c3, "%"), (97, c6, "%")):
            txt = "–" if val is None else (f"{val:,.0f}{unit}" if unit == " ms" else f"{val:.1f}{unit}")
            ax.text(x, y, txt, fontsize=11, ha="right", va="center", color=MUTED if val is None else ink,
                    fontweight=weight)
    fig.text(0.04, 0.025, "† turn-1-mini as published by P99Lab on the same benchmark (two views reported). "
             "– = not reported. Cue v1 = fine-tuned Gemma 4 E4B logits (words, context) × √Smart Turn v3.2 (voice).",
             fontsize=9, color=INK2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def heatmap(metric: str, title: str, sub: str, fmt, path: Path, lo: float, hi: float, legend_label: str = "best in the row") -> None:
    langs = ["en", "pt", "es", "fr", "de", "it", "nl", "ja", "ko", "zh", "id", "hi", "tr", "ar"]
    models = [CUE, "LiveKit Turn Detector v1", "Soniox", "Deepgram Flux", "Gradium", "ultraVAD",
              "LiveKit Turn Detector v1-mini", "SmartTurn v3.2", "OpenAI GPT Realtime 2", "AssemblyAI", "VAP (silent agent)"]
    models = [m for m in models if any(V[metric][l].get(m) is not None for l in langs)]
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    nr, nc = len(langs), len(models)
    fig = plt.figure(figsize=(14, 9.6), dpi=200)
    ax = fig.add_axes([0.11, 0.07, 0.86, 0.68])
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.axis("off")
    fig.text(0.03, 0.935, title, fontsize=21, fontweight="bold")
    fig.text(0.03, 0.895, sub, fontsize=11.5, color=INK2)
    for j, m in enumerate(models):
        label = NAMES.get(m, m)
        ax.text(j + 0.5, -0.25, label, rotation=35, ha="left", va="bottom", fontsize=10.5,
                color=BLUE if m == CUE else INK, fontweight="bold" if m == CUE else "normal",
                rotation_mode="anchor")
    for i, l in enumerate(langs):
        ax.text(-0.15, i + 0.5, LANG[l], ha="right", va="center", fontsize=11.5, color=INK)
        vals = {m: V[metric][l].get(m) for m in models}
        present = {m: v for m, v in vals.items() if v is not None}
        best = min(present, key=present.get)
        board = {k: x for k, x in V[metric][l].items() if x is not None and k not in ("VAD baseline", "Shortwire cue-v1 (Gemma)", "Shortwire cue-v1 (Jev)")}
        order = sorted(board, key=board.get)
        for j, m in enumerate(models):
            v = vals[m]
            g = 0.035
            if v is None:
                ax.add_patch(Rectangle((j + g, i + g), 1 - 2 * g, 1 - 2 * g, fc="#efece4", ec="none"))
                continue
            t = float(min(1.0, max(0.0, (v - lo) / (hi - lo))))
            c = cmap(t)
            ax.add_patch(FancyBboxPatch((j + g, i + g), 1 - 2 * g, 1 - 2 * g, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=c, ec="none"))
            dark = t > 0.55
            txt = fmt(v)
            ax.text(j + 0.5, i + 0.5 - (0.1 if m == CUE else 0), txt, ha="center", va="center", fontsize=11,
                    color="white" if dark else INK, fontweight="bold" if (m == best or m == CUE) else "normal")
            if m == CUE:
                r = order.index(m) + 1
                ax.text(j + 0.5, i + 0.78, {1: "1st", 2: "2nd", 3: "3rd"}.get(r, f"{r}th"), ha="center", va="center",
                        fontsize=8.5, color="white" if dark else INK2)
            if m == best:
                ax.add_patch(Rectangle((j + 0.07, i + 0.07), 0.86, 0.86, fc="none", ec=INK, lw=1.6))
    ax.add_patch(FancyBboxPatch((0.0, -0.02), 1.0, nr + 0.04, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="none", ec=BLUE, lw=2.4))
    # legend: ramp + best marker
    lx = fig.add_axes([0.62, 0.036, 0.2, 0.012])
    lx.imshow([[i / 100 for i in range(101)]], cmap=cmap, aspect="auto")
    lx.set_xticks([0, 100])
    lx.set_xticklabels([fmt(lo), fmt(hi) + "+"], fontsize=9, color=INK2)
    lx.set_yticks([])
    for sp in lx.spines.values():
        sp.set_visible(False)
    lx.tick_params(length=0)
    bx = fig.add_axes([0.875, 0.028, 0.014, 0.022]); bx.axis("off")
    bx.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fc="none", ec=INK, lw=1.6))
    fig.text(0.893, 0.033, legend_label, fontsize=10, color=INK)
    fig.text(0.11, 0.045, "Cue v1 = fine-tuned Gemma 4 E4B logits (words and context) × √Smart Turn v3.2 (voice). One English prompt for every language.", fontsize=9.5, color=INK2)
    fig.text(0.11, 0.022, "Rank under Cue counts every model eot-bench reports for that language. Grey: not covered.", fontsize=9.5, color=INK2)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
english_table(OUT / "cue-v1-english.png")
heatmap(L5, "Latency at a 5% false cut-off rate, 14 languages",
        "How long each model waits after the caller stops, when at most 5% of turns get cut off. Lighter is faster.",
        lambda v: f"{v * 1000:.0f}", OUT / "cue-v1-languages-latency.png", 0.3, 1.15, "fastest in the row")
heatmap(C3, "False cut-offs at a 300 ms latency budget, 14 languages",
        "Share of turns cut off when the model must answer within 300 ms. Lighter is better.",
        lambda v: f"{v * 100:.0f}%", OUT / "cue-v1-languages-cutoffs.png", 0.05, 0.55, "fewest in the row")
heatmap(L10, "Latency at a 10% false cut-off rate, 14 languages",
        "How long each model waits after the caller stops, when at most 10% of turns get cut off. Lighter is faster.",
        lambda v: f"{v * 1000:.0f}", OUT / "cue-v1-languages-latency-10pct.png", 0.2, 0.9, "fastest in the row")
heatmap(C6, "False cut-offs at a 600 ms latency budget, 14 languages",
        "Share of turns cut off when the model must answer within 600 ms. Lighter is better.",
        lambda v: f"{v * 100:.0f}%", OUT / "cue-v1-languages-cutoffs-600ms.png", 0.03, 0.2, "fewest in the row")
print("ok")
