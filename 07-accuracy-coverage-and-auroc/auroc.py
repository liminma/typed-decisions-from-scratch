"""AUROC: does the confidence put right answers above wrong ones?

    uv run 07-accuracy-coverage-and-auroc/auroc.py                  # BANKING77 test
    uv run 07-accuracy-coverage-and-auroc/auroc.py --dataset clinc150

AUROC is the probability that a randomly picked right answer has a higher confidence than a
randomly picked wrong one (a tie counts as half). 1.0 separates them perfectly; 0.5 is a coin
flip. The script computes it three ways, which agree, and then shows which changes to the
confidence leave it alone.
"""

import argparse

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from ece import load_answers
from temperature import Letters, fit, scaled_confidence


def auroc_by_pairs(scores: list[float], positive: list[bool]) -> float:
    """Count every (positive, negative) pair: 1 if the positive scores higher, 0.5 for a tie."""
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auroc_by_area(scores: list[float], positive: list[bool]) -> float:
    """The area under the ROC curve: lower the threshold step by step, plot the share of positives
    caught (true positive rate) against the share of negatives let through (false positive rate),
    and add up the area with trapezoids."""
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    by_score = {}
    for s, p in zip(scores, positive):
        tp, fp = by_score.get(s, (0, 0))
        by_score[s] = (tp + p, fp + (not p))
    tp = fp = 0
    area = 0.0
    for s in sorted(by_score, reverse=True):  # all answers with the same score pass together
        dtp, dfp = by_score[s]
        area += dfp / n_neg * (tp + dtp / 2) / n_pos  # trapezoid: width x mean height
        tp, fp = tp + dtp, fp + dfp
    return area


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    args = ap.parse_args()

    calib, test = load_answers(args.dataset, "calib"), load_answers(args.dataset, "test")
    conf = [a["confidence"] for a in test]
    correct = [a["correct"] for a in test]
    n_right = sum(correct)
    print(f"{args.dataset} test: {n_right} right and {len(test) - n_right} wrong answers, "
          f"{n_right * (len(test) - n_right):,} (right, wrong) pairs\n")

    print("1. Three ways to compute it, on the raw confidence:")
    print(f"   counting pairs:           {auroc_by_pairs(conf, correct):.4f}")
    print(f"   area under the ROC curve: {auroc_by_area(conf, correct):.4f}")
    print(f"   sklearn roc_auc_score:    {roc_auc_score(correct, conf):.4f}\n")

    # AUROC only looks at the order of the confidences, not their values
    letters = Letters(args.dataset)
    t_group, t_intent = fit(calib, letters)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        [a["confidence"] for a in calib], [float(a["correct"]) for a in calib])
    versions = {
        "raw": conf,
        "raw cubed (same order, other values)": [c ** 3 for c in conf],
        "temperature-scaled": [scaled_confidence(a, letters, t_group, t_intent) for a in test],
        "isotonic": list(iso.predict(conf)),
    }
    print("2. Changing the confidence:")
    for name, scores in versions.items():
        print(f"   {name:38} AUROC {roc_auc_score(correct, scores):.4f}   ({len(set(scores))} distinct values)")
    print("\n   Cubing keeps the order exactly, so AUROC doesn't move. Temperature scaling can reorder")
    print("   answers slightly: a scaled confidence depends on the whole distribution, not only the top")
    print("   probability. Isotonic never reorders, but it merges answers into ties, and a tied pair")
    print("   only counts half.")
