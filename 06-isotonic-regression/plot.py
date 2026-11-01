"""Two figures: the fitted staircase, and the test reliability diagram for all three confidences.

    uv run 06-isotonic-regression/plot.py
    uv run 06-isotonic-regression/plot.py --dataset clinc150

Saves results/staircase-<dataset>.png and results/reliability-<dataset>.png in this folder.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

from ece import ece, load_answers, reliability_bins
from temperature import Letters, fit, scaled_confidence

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAW, TEMPERATURE, ISOTONIC = "#2a78d6", "#eb6834", "#1baf7a"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="banking77")
args = ap.parse_args()
out_dir = Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

calib, test = load_answers(args.dataset, "calib"), load_answers(args.dataset, "test")
cx, cy = [a["confidence"] for a in calib], [float(a["correct"]) for a in calib]
iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(cx, cy)

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def axes(title: str):
    fig, ax = plt.subplots(figsize=(6, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return fig, ax


# 1. The staircase, over the calibration answers it was fitted to (ticks: right at the top, wrong at the bottom)
fig, ax = axes(f"{args.dataset.upper()}: the isotonic fit is a staircase\n"
               f"fitted on {len(calib)} calibration answers")
ax.plot(iso.X_thresholds_, iso.y_thresholds_, color=ISOTONIC, linewidth=2, label="isotonic fit")
ax.scatter([x for x, y in zip(cx, cy) if y == 1], [1.02] * int(sum(cy)), marker="|", s=60,
           color=MUTED, alpha=0.4, linewidths=1, label="calibration answers: right (top), wrong (bottom)")
ax.scatter([x for x, y in zip(cx, cy) if y == 0], [-0.02] * int(len(cy) - sum(cy)), marker="|", s=60,
           color=MUTED, alpha=0.4, linewidths=1)
ax.set_xlabel("Raw confidence")
ax.set_ylabel("Calibrated confidence")
ax.legend(loc="lower right", frameon=False, labelcolor=INK, bbox_to_anchor=(1, 0.04))
fig.savefig(out_dir / f"staircase-{args.dataset}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)

# 2. Reliability on the test answers: raw, temperature-scaled, isotonic
letters = Letters(args.dataset)
t_group, t_intent = fit(calib, letters)
correct = [a["correct"] for a in test]
methods = {
    "raw": ([a["confidence"] for a in test], RAW),
    "temperature": ([scaled_confidence(a, letters, t_group, t_intent) for a in test], TEMPERATURE),
    "isotonic": (list(iso.predict([a["confidence"] for a in test])), ISOTONIC),
}
fig, ax = axes(f"{args.dataset.upper()} test: raw, temperature and isotonic\n"
               f"{len(test):,} answers, 15 bins; marker area = answers in the bin")
for name, (conf, color) in methods.items():
    bins = reliability_bins(conf, correct)
    # Isotonic leaves a few answers on the ramps between its steps; sizing markers by count keeps
    # those small bins from looking as important as the big ones
    ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], color=color, linewidth=1, alpha=0.6)
    ax.scatter([b["conf"] for b in bins], [b["acc"] for b in bins], s=[8 + 0.35 * b["n"] for b in bins],
               color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=f"{name}, ECE {ece(conf, correct):.3f}")
ax.set_xlabel("Confidence of the chosen answer")
ax.set_ylabel("Accuracy in the bin")
ax.legend(loc="lower right", frameon=False, labelcolor=INK, markerscale=0.5)
fig.savefig(out_dir / f"reliability-{args.dataset}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"saved {out_dir}/staircase-{args.dataset}.png and reliability-{args.dataset}.png")
