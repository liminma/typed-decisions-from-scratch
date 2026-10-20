"""Draw the reliability diagram: accuracy against confidence, bin by bin.

    uv run 04-is-the-confidence-honest/plot.py                  # BANKING77 test
    uv run 04-is-the-confidence-honest/plot.py --split calib

Saves results/reliability-<dataset>-<split>.png in this folder. Needs matplotlib (uv sync).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ece import ece, load_answers, reliability_bins

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = "#2a78d6"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="banking77")
ap.add_argument("--split", default="test", choices=("calib", "test"))
args = ap.parse_args()

answers = load_answers(args.dataset, args.split)
conf = [a["confidence"] for a in answers]
correct = [a["correct"] for a in answers]
bins = reliability_bins(conf, correct)

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 6.4), sharex=True, facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
for ax in (top, bottom):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# Top: each bin's accuracy at its mean confidence. On the diagonal, the confidence is honest.
top.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
top.text(0.62, 0.55, "confidence = accuracy", color=MUTED, rotation=45, ha="center", va="center", fontsize=9)
for b in bins:  # the gap the ECE adds up
    top.plot([b["conf"], b["conf"]], [b["conf"], b["acc"]], color=SERIES, linewidth=2, alpha=0.35,
             solid_capstyle="round")
top.plot([b["conf"] for b in bins], [b["acc"] for b in bins], color=SERIES, linewidth=2,
         marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
top.set_xlim(0, 1)
top.set_ylim(-0.03, 1.03)
top.set_ylabel("Accuracy in the bin")
# Overall, were the answers right more or less often than the model said?
verdict = "more" if sum(correct) / len(correct) > sum(conf) / len(conf) else "less"
top.set_title(f"{args.dataset.upper()} {args.split}: right {verdict} often than the model says\n"
              f"{len(answers):,} answers, 15 bins, ECE {ece(conf, correct):.3f}",
              loc="left", fontsize=11, color=INK)

# Bottom: how many answers each bin holds, since a bin of 5 says little
width = 1 / 15
bottom.bar([b["lo"] + width / 2 for b in bins], [b["n"] for b in bins], width=width * 0.85,
           color=SERIES, edgecolor=SURFACE, linewidth=0)
bottom.set_ylabel("Answers")
bottom.set_xlabel("Confidence of the chosen answer")

out = Path(__file__).parent / "results" / f"reliability-{args.dataset}-{args.split}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"saved {out}")
