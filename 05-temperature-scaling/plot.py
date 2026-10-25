"""Reliability diagram before and after temperature scaling, on the test answers.

    uv run 05-temperature-scaling/plot.py
    uv run 05-temperature-scaling/plot.py --dataset clinc150

Saves results/temperature-<dataset>.png in this folder. Needs matplotlib (uv sync).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ece import ece, load_answers, reliability_bins
from temperature import Letters, fit, scaled_confidence

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAW, SCALED = "#2a78d6", "#eb6834"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="banking77")
args = ap.parse_args()

letters = Letters(args.dataset)
t_group, t_intent = fit(load_answers(args.dataset, "calib"), letters)
test = load_answers(args.dataset, "test")
correct = [a["correct"] for a in test]
series = {
    "raw": ([a["confidence"] for a in test], RAW),
    "temperature-scaled": ([scaled_confidence(a, letters, t_group, t_intent) for a in test], SCALED),
}

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
fig, ax = plt.subplots(figsize=(6, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax.text(0.62, 0.55, "confidence = accuracy", color=MUTED, rotation=45, ha="center", va="center", fontsize=9)
for name, (conf, color) in series.items():
    bins = reliability_bins(conf, correct)
    ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], color=color, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
            label=f"{name}, ECE {ece(conf, correct):.3f}")

ax.set_xlim(0, 1)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Confidence of the chosen answer (15 bins)")
ax.set_ylabel("Accuracy in the bin")
ax.legend(loc="upper left", frameon=False, labelcolor=INK)
ax.set_title(f"{args.dataset.upper()} test: before and after temperature scaling\n"
             f"T fitted on the calibration answers: group {t_group:.2f}, intent {t_intent:.2f}",
             loc="left", fontsize=11, color=INK)

out = Path(__file__).parent / "results" / f"temperature-{args.dataset}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"saved {out}")
