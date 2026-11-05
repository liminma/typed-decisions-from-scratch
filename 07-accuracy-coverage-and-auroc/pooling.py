"""The pooling trap: what one AUROC over two kinds of traffic is made of.

    uv run 07-accuracy-coverage-and-auroc/pooling.py
    uv run 07-accuracy-coverage-and-auroc/pooling.py --scaled     # temperature-scaled confidence

Pool the BANKING77 and CLINC150 test answers and compute one AUROC of right against wrong. Its
(right, wrong) pairs are of four kinds: both from BANKING77, both from CLINC150, or one from each.
The pooled AUROC is the average of the four kinds' AUROCs, weighted by how many pairs each has.
So it rises above both datasets' own AUROCs whenever the cross pairs are easier than the pairs
inside each dataset. That happens when one kind of traffic is both more accurate and more
confident than the other: its right answers then easily outrank the other's wrong ones. Needs both
datasets' answers from 03-where-confidence-comes-from.
"""

import argparse
import bisect

from sklearn.metrics import roc_auc_score

from ece import load_answers
from temperature import Letters, fit, scaled_confidence


def pair_auroc(right: list[float], wrong: list[float]) -> float:
    """Share of (right, wrong) pairs where the right answer is more confident; ties count half."""
    right = sorted(right)
    wins = 0.0
    for w in wrong:
        lo, hi = bisect.bisect_left(right, w), bisect.bisect_right(right, w)
        wins += (len(right) - hi) + 0.5 * (hi - lo)
    return wins / (len(right) * len(wrong))


ap = argparse.ArgumentParser()
ap.add_argument("--scaled", action="store_true", help="temperature-scale each dataset's confidence (post 5)")
args = ap.parse_args()

answers = {d: load_answers(d, "test") for d in ("banking77", "clinc150")}
conf = {}
for d, ans in answers.items():
    if args.scaled:  # each dataset with its own temperatures, fitted on its own calibration answers
        letters = Letters(d)
        t_group, t_intent = fit(load_answers(d, "calib"), letters)
        conf[d] = [scaled_confidence(a, letters, t_group, t_intent) for a in ans]
    else:
        conf[d] = [a["confidence"] for a in ans]
right = {d: [c for c, a in zip(conf[d], ans) if a["correct"]] for d, ans in answers.items()}
wrong = {d: [c for c, a in zip(conf[d], ans) if not a["correct"]] for d, ans in answers.items()}

kind = "temperature-scaled" if args.scaled else "raw"
print(f"The four kinds of (right, wrong) pairs in the pooled test answers, {kind} confidence:")
total_pairs, weighted = 0, 0.0
for r in answers:
    for w in answers:
        n = len(right[r]) * len(wrong[w])
        auc = pair_auroc(right[r], wrong[w])
        total_pairs += n
        weighted += n * auc
        print(f"  right from {r:9}  vs  wrong from {w:9}  {n:>10,} pairs   AUROC {auc:.3f}")
print(f"\n  weighted average: {weighted / total_pairs:.3f}")

pooled = [c for d in answers for c in conf[d]]
correct = [a["correct"] for ans in answers.values() for a in ans]
print(f"  pooled AUROC:     {roc_auc_score(correct, pooled):.3f}  (the same number)\n")
print("The pooled AUROC is pulled up when the cross pairs are easier than the pairs inside each")
print("dataset, and it can land above both. Report each kind of traffic's AUROC next to the pooled one.")
