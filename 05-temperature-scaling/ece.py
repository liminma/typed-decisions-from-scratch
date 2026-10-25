"""Is the confidence honest? Bin the answers by confidence, compare with accuracy, and sum the gaps.

    python3 05-temperature-scaling/ece.py                  # BANKING77 test
    python3 05-temperature-scaling/ece.py --split calib

Reads the answers saved by 03-where-confidence-comes-from/run.py. Standard library only.

ECE (expected calibration error), as used throughout the series:
  - 15 bins of equal width over 0-1, by the confidence of the chosen answer;
  - in each non-empty bin, the gap |mean confidence - accuracy|;
  - weighted by the bin's share of the answers, and summed.
"""

import argparse
import bisect
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "03-where-confidence-comes-from" / "results"


def load_answers(dataset: str, split: str) -> list[dict]:
    path = RESULTS / f"{dataset}-{split}.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found. Run 03-where-confidence-comes-from/run.py --split {split} first.")
    return [json.loads(line) for line in path.open()]


def reliability_bins(conf: list[float], correct: list[bool], n_bins: int = 15) -> list[dict]:
    """One entry per non-empty bin: its range, size, mean confidence and accuracy.
    Bin b holds confidences in [b/n, (b+1)/n); a confidence of exactly 1.0 goes in the last bin."""
    edges = [b * (1 / n_bins) for b in range(n_bins)] + [1.0]
    members = [[] for _ in range(n_bins)]
    for c, ok in zip(conf, correct):
        members[min(bisect.bisect_right(edges[1:-1], c), n_bins - 1)].append((c, ok))
    return [{"lo": edges[b], "hi": edges[b + 1], "n": len(m),
             "conf": sum(c for c, _ in m) / len(m), "acc": sum(ok for _, ok in m) / len(m)}
            for b, m in enumerate(members) if m]


def ece(conf: list[float], correct: list[bool], n_bins: int = 15) -> float:
    total = len(conf)
    return sum(abs(b["conf"] - b["acc"]) * b["n"] / total for b in reliability_bins(conf, correct, n_bins))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    ap.add_argument("--split", default="test", choices=("calib", "test"))
    args = ap.parse_args()

    answers = load_answers(args.dataset, args.split)
    conf = [a["confidence"] for a in answers]
    correct = [a["correct"] for a in answers]
    print(f"{args.dataset} {args.split}: {len(answers)} answers, accuracy {sum(correct) / len(correct):.1%}, "
          f"mean confidence {sum(conf) / len(conf):.1%}\n")

    print(f"{'bin':>13} {'answers':>8} {'confidence':>11} {'accuracy':>9} {'gap':>7} {'adds to ECE':>12}")
    for b in reliability_bins(conf, correct):
        gap = b["acc"] - b["conf"]
        print(f"{b['lo']:.3f}-{b['hi']:.3f} {b['n']:8} {b['conf']:11.3f} {b['acc']:9.3f} {gap:+7.3f} "
              f"{abs(gap) * b['n'] / len(conf):12.4f}")
    print(f"\nECE, 15 bins: {ece(conf, correct):.4f}")
    print("A positive gap means the answers were right more often than the model said (under-confident).")

    # The number depends on the binning, so always say which one you used
    print("\nThe same answers with other bin counts:")
    for n in (5, 10, 15, 20, 30):
        print(f"  {n:2} bins: ECE {ece(conf, correct, n):.4f}")

    # A low ECE doesn't mean the confidence is useful: one number for every answer, equal to the
    # overall accuracy, is perfectly calibrated and tells right from wrong not at all (post 7).
    flat = [sum(correct) / len(correct)] * len(conf)
    print(f"\nEvery answer given the same confidence, {flat[0]:.3f} (the accuracy): ECE {ece(flat, correct):.4f}")
