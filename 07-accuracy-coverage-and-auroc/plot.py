"""Two figures: BANKING77's accuracy-coverage curve, and the ROC curve for CLINC150's out-of-scope requests.

    uv run 07-accuracy-coverage-and-auroc/plot.py

Saves results/coverage-banking77.png and results/roc-clinc150-oos.png in this folder. The second
needs the CLINC150 answers from 03-where-confidence-comes-from/run.py --dataset clinc150.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from coverage import accuracy_at_coverage
from ece import RESULTS, load_answers
from temperature import Letters

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = "#2a78d6"
out_dir = Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def axes(title: str):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return fig, ax


# 1. Accuracy against coverage, BANKING77 test
test = load_answers("banking77", "test")
conf, correct = [a["confidence"] for a in test], [a["correct"] for a in test]
coverages = [c / 100 for c in range(5, 101)]
accuracy = [accuracy_at_coverage(conf, correct, c) for c in coverages]
at80 = accuracy_at_coverage(conf, correct, 0.8)
fig, ax = axes(f"BANKING77 test: answering only when confident\n"
               f"accuracy of the most confident answers, {len(test):,} in all")
ax.plot([c * 100 for c in coverages], [a * 100 for a in accuracy], color=SERIES, linewidth=2)
for c, a, offset in ((1.0, accuracy[-1], (-10, 4)), (0.8, at80, (-8, -16))):
    ax.scatter([c * 100], [a * 100], s=60, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(f"{a:.1%} at {c:.0%}", (c * 100, a * 100), xytext=offset, textcoords="offset points",
                ha="right", color=INK, fontsize=9)
ax.set_xlim(0, 102)
ax.set_xlabel("Coverage: share of answers kept (%)")
ax.set_ylabel("Accuracy of the kept answers (%)")
fig.savefig(out_dir / "coverage-banking77.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"saved {out_dir / 'coverage-banking77.png'}")

# 2. ROC curve: P(OUT_OF_SCOPE) as a detector of out-of-scope requests, CLINC150 test
if not (RESULTS / "clinc150-test.jsonl").exists():
    raise SystemExit("No CLINC150 answers yet: run 03-where-confidence-comes-from/run.py --dataset clinc150")
oos = Letters("clinc150").code["oos"][0]
test = load_answers("clinc150", "test")
p_oos = [a["p_group"].get(oos, 0.0) for a in test]
is_oos = [a["gold"] == "oos" for a in test]
fpr, tpr, thresholds = roc_curve(is_oos, p_oos)
fig, ax = axes(f"CLINC150 test: P(OUT_OF_SCOPE) as a detector\n"
               f"{sum(is_oos):,} out-of-scope vs {len(is_oos) - sum(is_oos):,} in-scope requests, "
               f"AUROC {roc_auc_score(is_oos, p_oos):.3f}")
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax.text(0.6, 0.53, "chance", color=MUTED, rotation=38, ha="center", fontsize=9)
ax.plot(fpr, tpr, color=SERIES, linewidth=2)
n_oos, n_in = sum(is_oos), len(is_oos) - sum(is_oos)
for t in (0.9, 0.5, 0.1):
    x = sum(p >= t for p, o in zip(p_oos, is_oos) if not o) / n_in
    y = sum(p >= t for p, o in zip(p_oos, is_oos) if o) / n_oos
    ax.scatter([x], [y], s=50, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(f"t = {t}", (x, y), xytext=(8, -12), textcoords="offset points", color=INK, fontsize=9)
ax.set_xlim(-0.01, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("In-scope requests flagged by mistake (false positive rate)")
ax.set_ylabel("Out-of-scope requests caught (true positive rate)")
fig.savefig(out_dir / "roc-clinc150-oos.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"saved {out_dir / 'roc-clinc150-oos.png'}")
