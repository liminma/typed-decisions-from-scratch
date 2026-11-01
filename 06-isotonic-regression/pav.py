"""Isotonic regression from scratch: pool adjacent violators.

    uv run 06-isotonic-regression/pav.py

Given (raw confidence, right or wrong) pairs, find the non-decreasing function f with the smallest
squared error sum (f(x) - y)^2. Then f(x) estimates how often answers with raw confidence x are
right. The algorithm:

  1. sort the pairs by x; each starts as its own block, valued at its y (1 if right, 0 if wrong);
  2. whenever a block is valued higher than the block after it, merge the two into one block,
     valued at the mean of all its y;
  3. repeat until the values only go up.

At run time, f is flat inside each block and a straight ramp between neighbouring blocks.
Standard library only; the check at the end compares it with scikit-learn.
"""


def fit_pav(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """The fitted f as breakpoints (xs, ys): each block's first and last x, at the block's value."""
    # Equal x values can't be ordered, so they start as one block
    points = {}
    for xi, yi in zip(x, y):
        s, n = points.get(xi, (0.0, 0))
        points[xi] = (s + yi, n + 1)

    blocks = []  # each block: [sum of y, count, first x, last x]
    for xi in sorted(points):
        s, n = points[xi]
        blocks.append([s, n, xi, xi])
        # Merge backwards while the previous block is valued higher than this one
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n, _, last = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
            blocks[-1][3] = last

    xs, ys = [], []
    for s, n, first, last in blocks:
        for xb in ([first] if first == last else [first, last]):
            xs.append(xb)
            ys.append(s / n)
    return xs, ys


def predict(xs: list[float], ys: list[float], x: float) -> float:
    """f(x): flat beyond the ends, straight lines between breakpoints (like numpy's interp)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = max(i for i in range(len(xs)) if xs[i] <= x)
    if xs[lo] == x or lo == len(xs) - 1:
        return ys[lo]
    return ys[lo] + (ys[lo + 1] - ys[lo]) * (x - xs[lo]) / (xs[lo + 1] - xs[lo])


if __name__ == "__main__":
    # A small example: six answers, sorted by raw confidence
    x = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    y = [0, 1, 0, 1, 1, 1]
    xs, ys = fit_pav(x, y)
    print("Raw confidence: " + "  ".join(f"{v:.2f}" for v in x))
    print("Right?          " + "  ".join(f"{v:4}" for v in y))
    print("Calibrated:     " + "  ".join(f"{predict(xs, ys, v):.2f}" for v in x))
    print("0.40 (right) and 0.50 (wrong) were out of order, so they were merged into one block at 0.5.\n")

    # The same on the BANKING77 calibration answers, checked against scikit-learn
    from sklearn.isotonic import IsotonicRegression

    from ece import load_answers

    calib, test = load_answers("banking77", "calib"), load_answers("banking77", "test")
    cx, cy = [a["confidence"] for a in calib], [float(a["correct"]) for a in calib]
    xs, ys = fit_pav(cx, cy)
    sk = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(cx, cy)
    tx = [a["confidence"] for a in test]
    worst = max(abs(predict(xs, ys, v) - s) for v, s in zip(tx, sk.predict(tx)))
    print(f"BANKING77: fitted on {len(cx)} calibration answers, {len(set(ys))} distinct values.")
    print(f"Largest difference from scikit-learn over the {len(tx)} test answers: {worst:.1e}")
