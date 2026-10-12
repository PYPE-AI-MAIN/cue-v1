"""One portrait infographic (4:5, 1080x1350 at 2x) for the Cue v1 LinkedIn post: headline numbers, the English
table and the 14-language heatmap, drawn from eot-bench's results (the release leaderboard's summary.json).

    python make_linkedin.py <summary.json> <out.png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

s = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])
V = s["values"]
L5, C3 = "best_mean_latency_at_5pct_cutoff", "best_cutoff_rate_at_0_3s_latency"
CUE = "Shortwire Cue v1 (fine-tuned)"
SKIP = ("VAD baseline", "Shortwire cue-v1 (Gemma)", "Shortwire cue-v1 (Jev)")

SURFACE, CARD = "#faf8f3", "#f1eee6"
INK, INK2, MUTED, RULE = "#1b1b1a", "#5f5e5a", "#9a9890", "#e6e3da"
BLUE, BLUE_TINT = "#2a78d6", "#e6f0fb"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("blue", RAMP)
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial"], "text.color": INK})
NAMES = {CUE: "Cue v1", "LiveKit Turn Detector v1": "LiveKit v1", "LiveKit Turn Detector v1-mini": "LiveKit v1-mini",
         "SmartTurn v3.2": "Smart Turn v3.2", "OpenAI GPT Realtime 2": "OpenAI Realtime 2", "JoinIn AI Baton": "Baton",
         "VAP (silent agent)": "VAP"}
LANG = {"en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
        "nl": "Dutch", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "id": "Indonesian", "hi": "Hindi",
        "tr": "Turkish", "ar": "Arabic"}


def faster(rival):
    w = t = 0
    for m in (L5, "best_mean_latency_at_10pct_cutoff"):
        for l in s["languages"]:
            a, b = V[m][l].get(CUE), V[m][l].get(rival)
            if a is not None and b is not None:
                t, w = t + 1, w + (a < b)
    return w, t


fig = plt.figure(figsize=(10.8, 13.5), dpi=200)
fig.patch.set_facecolor(SURFACE)

# --- headline --------------------------------------------------------------------------------------------------
fig.text(0.06, 0.955, "SHORTWIRE  ·  CUE V1", fontsize=13, color=BLUE, fontweight="bold")
fig.text(0.06, 0.918, "Turn detection that knows when to wait", fontsize=29, fontweight="bold")
fig.text(0.06, 0.893, "Context from fine-tuned Gemma logits plus Smart Turn for the voice, on LiveKit's eot-bench. "
         "14 languages.", fontsize=12.5, color=INK2)

# --- four numbers ----------------------------------------------------------------------------------------------
st_w, st_t = faster("SmartTurn v3.2")
uv_w, uv_t = faster("ultraVAD")
tiles = [("1st", "in Portuguese", "681 ms, ahead of LiveKit v1\nand Soniox at 702 ms"),
         ("4th", "in English", "719 ms, ahead of ultraVAD,\nSmart Turn and OpenAI Realtime"),
         (f"{st_w}/{st_t}", "faster than Smart Turn alone", "14 languages × the 5% and\n10% cut-off budgets"),
         (f"{uv_w}/{uv_t}", "faster than ultraVAD", "same benchmark, same\nharness, same 14 languages")]
tw, th, gap, y0 = 0.205, 0.105, 0.0167, 0.765
for i, (big, what, note) in enumerate(tiles):
    x = 0.06 + i * (tw + gap)
    ax = fig.add_axes([x, y0, tw, th])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", fc=CARD, ec="none",
                                transform=ax.transAxes))
    ax.text(0.09, 0.70, big, fontsize=30, fontweight="bold", color=BLUE, transform=ax.transAxes, va="center")
    ax.text(0.09, 0.43, what, fontsize=11, fontweight="bold", transform=ax.transAxes, va="center")
    ax.text(0.09, 0.19, note, fontsize=8.6, color=INK2, transform=ax.transAxes, va="center", linespacing=1.3)

# --- English table ---------------------------------------------------------------------------------------------
fig.text(0.06, 0.728, "English: latency at a 5% false cut-off rate", fontsize=15, fontweight="bold")
fig.text(0.06, 0.711, "How long the agent waits after the caller stops, with at most 5% of turns cut off. Lower is better. "
         "Selected models; rank is across all 14 on the board.",
         fontsize=10, color=INK2)
show = [CUE, "LiveKit Turn Detector v1", "JoinIn AI Baton", "Soniox", "ultraVAD", "SmartTurn v3.2",
        "LiveKit Turn Detector v1-mini", "OpenAI GPT Realtime 2"]
board = {m: v for m, v in V[L5]["en"].items() if v is not None and m not in SKIP}
board["turn-1-mini"] = 0.849                                    # as published by P99Lab on the same benchmark
cut = dict(V[C3]["en"]); cut["turn-1-mini"] = 0.226
order = sorted(board, key=board.get)
rows = sorted([m for m in show + ["turn-1-mini"]], key=board.get)
ax = fig.add_axes([0.06, 0.445, 0.88, 0.255])
ax.set_xlim(0, 100)
ax.set_ylim(len(rows), -0.7)
ax.axis("off")
for x, label, ha in ((0, "Rank", "left"), (9, "Model", "left"), (35, "Latency @ 5% cut-offs", "left"),
                     (100, "Cut-offs @ 300 ms", "right")):
    ax.text(x, -0.35, label, fontsize=9.5, color=INK2, fontweight="bold", ha=ha, va="center")
ax.plot([0, 100], [-0.05, -0.05], color=INK, lw=0.8)
vmax = max(board[m] for m in rows)
for i, m in enumerate(rows):
    y, cue = i + 0.5, m == CUE
    if cue:
        ax.add_patch(FancyBboxPatch((-1, i + 0.06), 102, 0.88, boxstyle="round,pad=0,rounding_size=0.5", fc=BLUE_TINT,
                                    ec="none", zorder=0))
    else:
        ax.plot([0, 100], [i + 1, i + 1], color=RULE, lw=0.7)
    wt = "bold" if cue else "normal"
    ax.text(1, y, str(order.index(m) + 1), fontsize=11, color=INK2, va="center")
    name = NAMES.get(m, m) + (" †" if m == "turn-1-mini" else "")
    ax.text(9, y, name, fontsize=12, va="center", fontweight=wt, color=BLUE if cue else INK)
    w = 33 * board[m] / vmax
    ax.add_patch(FancyBboxPatch((35, y - 0.16), w, 0.32, boxstyle="round,pad=0,rounding_size=0.14",
                                fc=BLUE if cue else "#c9c6bc", ec="none"))
    ax.text(35 + w + 1, y, f"{board[m] * 1000:,.0f} ms", fontsize=11.5, va="center", fontweight=wt)
    c = cut.get(m)
    ax.text(100, y, "–" if c is None else f"{c * 100:.1f}%", fontsize=11.5, va="center", ha="right", fontweight=wt,
            color=MUTED if c is None else INK)

# --- 14-language heatmap ---------------------------------------------------------------------------------------
fig.text(0.06, 0.413, "14 languages, one model, one English prompt", fontsize=15, fontweight="bold")
fig.text(0.06, 0.396, "Latency at a 5% false cut-off rate, in ms. Lighter is faster; the boxed cell is the fastest in "
         "its row.", fontsize=10, color=INK2)
langs = ["en", "pt", "es", "fr", "de", "it", "nl", "ja", "ko", "zh", "id", "hi", "tr", "ar"]
cols = [CUE, "LiveKit Turn Detector v1", "Soniox", "ultraVAD", "SmartTurn v3.2", "LiveKit Turn Detector v1-mini"]
hx = fig.add_axes([0.17, 0.075, 0.77, 0.29])
hx.set_xlim(0, len(cols))
hx.set_ylim(len(langs), -0.85)
hx.axis("off")
lo, hi = 0.3, 1.15
for j, m in enumerate(cols):
    hx.text(j + 0.5, -0.42, NAMES.get(m, m), ha="center", va="center", fontsize=10.5,
            fontweight="bold" if m == CUE else "normal", color=BLUE if m == CUE else INK)
for i, l in enumerate(langs):
    hx.text(-0.1, i + 0.5, LANG[l], ha="right", va="center", fontsize=10.5)
    full = {m: v for m, v in V[L5][l].items() if v is not None and m not in SKIP}
    rank = sorted(full, key=full.get)
    best = rank[0]
    for j, m in enumerate(cols):
        v, g = V[L5][l].get(m), 0.04
        if v is None:
            hx.add_patch(Rectangle((j + g, i + g), 1 - 2 * g, 1 - 2 * g, fc="#efece4", ec="none"))
            continue
        t = float(min(1.0, max(0.0, (v - lo) / (hi - lo))))
        hx.add_patch(FancyBboxPatch((j + g, i + g), 1 - 2 * g, 1 - 2 * g, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=cmap(t), ec="none"))
        label = f"{v * 1000:.0f}"
        if m == CUE:
            r = rank.index(m) + 1
            label += f"  · {({1: '1st', 2: '2nd', 3: '3rd'}).get(r, f'{r}th')}"
        hx.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=10,
                color="white" if t > 0.55 else INK, fontweight="bold" if m in (CUE, best) else "normal")
        if m == best:
            hx.add_patch(Rectangle((j + 0.09, i + 0.1), 0.82, 0.8, fc="none", ec=INK, lw=1.5))
hx.add_patch(FancyBboxPatch((0.0, -0.02), 1.0, len(langs) + 0.04, boxstyle="round,pad=0,rounding_size=0.1",
                            fc="none", ec=BLUE, lw=2.2))

# --- footer ----------------------------------------------------------------------------------------------------
fig.text(0.06, 0.045, "Cue v1 = logits from Gemma 4 E4B fine-tuned on turn decisions (DONE, LIKELY DONE, MID-THOUGHT, HOLDING) "
         "× √Smart Turn v3.2: context and voice.", fontsize=8.8, color=INK2)
fig.text(0.06, 0.03, "Rank counts every model eot-bench reports for that language. † turn-1-mini as published by P99Lab "
         "on the same benchmark.", fontsize=8.8, color=INK2)
fig.text(0.06, 0.012, "Open weights · Apache 2.0 · Hugging Face: Pype/cue-v1", fontsize=9.5, color=INK, fontweight="bold")
fig.savefig(OUT, facecolor=SURFACE)
print("ok")
