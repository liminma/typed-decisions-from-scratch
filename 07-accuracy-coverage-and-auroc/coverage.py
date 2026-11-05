"""Accuracy against coverage: answer only when confident, and see how accuracy rises.

    python3 07-accuracy-coverage-and-auroc/coverage.py                  # BANKING77 test
    python3 07-accuracy-coverage-and-auroc/coverage.py --dataset clinc150

Sort the answers from most to least confident. Keep the top share (the coverage) and hand the
rest to someone else; the kept answers' accuracy is the selective accuracy at that coverage.
Standard library only.
"""

import argparse

from ece import load_answers


def accuracy_at_coverage(conf: list[float], correct: list[bool], coverage: float) -> float:
    """Accuracy on the most confident `coverage` share. Where the cut splits answers with the
    same confidence, the tied group counts at its average accuracy (as if their order were random)."""
    k = max(1, round(coverage * len(conf)))
    groups = {}
    for c, ok in zip(conf, correct):
        n, right = groups.get(c, (0, 0))
        groups[c] = (n + 1, right + ok)
    taken, right_taken = 0, 0.0
    for c in sorted(groups, reverse=True):
        n, right = groups[c]
        m = min(n, k - taken)
        right_taken += m * right / n
        taken += m
        if taken == k:
            break
    return right_taken / k


def threshold_at_coverage(conf: list[float], coverage: float) -> float:
    """The lowest confidence that is still kept at this coverage."""
    k = max(1, round(coverage * len(conf)))
    return sorted(conf, reverse=True)[k - 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    args = ap.parse_args()

    test = load_answers(args.dataset, "test")
    conf = [a["confidence"] for a in test]
    correct = [a["correct"] for a in test]
    print(f"{args.dataset} test: {len(test)} answers, raw confidence\n")
    print(f"  {'coverage':>8}  {'answers kept':>12}  {'confidence at least':>19}  {'accuracy':>8}")
    for pct in (100, 95, 90, 85, 80, 70, 60, 50, 40, 30, 20, 10):
        c = pct / 100
        print(f"  {pct:7}%  {round(c * len(conf)):12}  {threshold_at_coverage(conf, c):19.3f}  "
              f"{accuracy_at_coverage(conf, correct, c):8.1%}")
    print("\nHanding the least confident 20% to a person lifts accuracy on the rest from "
          f"{accuracy_at_coverage(conf, correct, 1.0):.1%} to {accuracy_at_coverage(conf, correct, 0.8):.1%}.")
