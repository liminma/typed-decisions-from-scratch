"""Isotonic regression on BANKING77, compared with temperature scaling.

    uv run 06-isotonic-regression/isotonic.py
    uv run 06-isotonic-regression/isotonic.py --dataset clinc150

Fits scikit-learn's IsotonicRegression on the calibration answers (raw confidence -> right or
wrong), prints the fitted staircase, and compares raw, temperature-scaled and isotonic confidence
on the test answers. pav.py shows what IsotonicRegression does inside.
"""

import argparse
from collections import Counter

from sklearn.isotonic import IsotonicRegression

from ece import ece, load_answers
from temperature import Letters, fit, scaled_confidence


def accuracy_at_coverage(conf: list[float], correct: list[bool], coverage: float) -> tuple[float, float, float]:
    """Accuracy on the most confident `coverage` share of the answers.

    When the cut falls inside a group of tied confidences, which of them make the cut is
    arbitrary. So this returns the expected accuracy over every order of the ties, then the
    worst and the best case."""
    k = max(1, round(coverage * len(conf)))
    groups = {}
    for c, ok in zip(conf, correct):
        n, right = groups.get(c, (0, 0))
        groups[c] = (n + 1, right + ok)
    taken, right_exp, right_min, right_max = 0, 0.0, 0, 0
    for c in sorted(groups, reverse=True):
        n, right = groups[c]
        m = min(n, k - taken)  # how many of this tied group make the cut
        right_exp += m * right / n
        right_min += max(0, m - (n - right))  # the wrong ones go first
        right_max += min(m, right)            # the right ones go first
        taken += m
        if taken == k:
            break
    return right_exp / k, right_min / k, right_max / k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    args = ap.parse_args()

    calib, test = load_answers(args.dataset, "calib"), load_answers(args.dataset, "test")
    raw_calib = [a["confidence"] for a in calib]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw_calib, [float(a["correct"]) for a in calib])

    # The staircase: each step is a block of calibration answers that share one value
    print(f"The fitted staircase ({len(calib)} calibration answers):")
    print(f"  {'raw confidence':>17}  {'answers':>7}  {'calibrated':>10}")
    steps = Counter(round(v, 6) for v in iso.predict(raw_calib))
    for value in sorted(steps):
        xs = [x for x, v in zip(raw_calib, iso.predict(raw_calib)) if round(v, 6) == value]
        print(f"  {min(xs):.3f} to {max(xs):.3f}  {steps[value]:7}  {value:10.3f}")
    print("  Between steps, the curve is a straight ramp from one value to the next.\n")

    letters = Letters(args.dataset)
    t_group, t_intent = fit(calib, letters)
    correct = [a["correct"] for a in test]
    methods = {
        "raw": [a["confidence"] for a in test],
        "temperature": [scaled_confidence(a, letters, t_group, t_intent) for a in test],
        "isotonic": list(iso.predict([a["confidence"] for a in test])),
    }
    print(f"Test answers ({len(test)}), all fitted on the calibration answers only:")
    print(f"  {'':12} {'ECE':>6} {'mean conf.':>10} {'distinct values':>15} {'largest tie':>12}   accuracy at 80% coverage")
    for name, conf in methods.items():
        tie = max(Counter(conf).values())
        exp, lo, hi = accuracy_at_coverage(conf, correct, 0.8)
        spread = f"  ({lo:.1%} to {hi:.1%}, by tie order)" if hi > lo else ""
        print(f"  {name:12} {ece(conf, correct):6.3f} {sum(conf) / len(conf):10.3f} {len(set(conf)):15} "
              f"{tie / len(conf):12.1%}   {exp:.1%}{spread}")

    # The top step promises certainty because every calibration answer in it was right
    sure = [ok for c, ok in zip(methods["isotonic"], correct) if c == 1.0]
    if sure:
        print(f"\nTest answers given exactly 1.000 by isotonic: {len(sure)}, of which {sum(sure) / len(sure):.1%} right.")
