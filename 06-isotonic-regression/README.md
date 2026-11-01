# 06 - When one knob isn't enough: isotonic regression

Code for Post 6: When one knob isn't enough: isotonic regression

A correction that can bend differently in different ranges, and what it costs.

These scripts don't call the server. They read the answers saved by [`03-where-confidence-comes-from/run.py`](../03-where-confidence-comes-from/), so run that first, for both splits. They need scikit-learn and matplotlib (`uv sync` at the repo root). `ece.py` and `temperature.py` are copies from [`05-temperature-scaling/`](../05-temperature-scaling/).

| Script | Shows |
|---|---|
| `pav.py` | Isotonic regression from scratch (pool adjacent violators), checked against scikit-learn |
| `isotonic.py` | The staircase fitted on BANKING77, and raw, temperature and isotonic compared on the test answers |
| `plot.py` | The staircase, and the test reliability diagram for all three |

## 1. Pool adjacent violators

```bash
uv run 06-isotonic-regression/pav.py
```

Isotonic regression takes the calibration answers as (raw confidence, right or wrong) pairs. It finds the curve that never goes down and is as close to them as possible. The curve's height at a raw confidence then estimates how often answers with that confidence are right.

The algorithm sorts the answers by confidence and merges neighbours that are out of order:

```
Raw confidence: 0.30  0.40  0.50  0.60  0.70  0.80
Right?             0     1     0     1     1     1
Calibrated:     0.00  0.50  0.50  1.00  1.00  1.00
```

At 0.40 the answer was right and at 0.50 it was wrong, which would make the curve go down. So the two are merged into one block, at their average, 0.5. The from-scratch version in `pav.py` gives the same predictions as scikit-learn's `IsotonicRegression` on every BANKING77 test answer.

## 2. The fit on BANKING77

```bash
uv run 06-isotonic-regression/isotonic.py
```

The 770 calibration answers give 11 steps. Yours may differ in number and position, because they come from your own post 3 answers:

```
     raw confidence  answers  calibrated
  0.102 to 0.115        2       0.000
  0.119 to 0.305       20       0.150
  ...
  0.561 to 0.787      123       0.797
  ...
  0.885 to 0.965      263       0.954
  0.965 to 0.991      200       1.000
```

Between steps, the curve rises in a straight line. Unlike a temperature, it can lift one range a lot (raw 0.48–0.56 becomes 0.75) and another hardly at all (raw 0.36–0.48 stays near 0.46).

On the test answers:

```
                  ECE mean conf. distinct values  largest tie   accuracy at 80% coverage
  raw           0.041      0.832            3079         0.1%   94.9%
  temperature   0.030      0.845            3079         0.1%   95.1%
  isotonic      0.011      0.878              66        32.4%   94.6%  (92.7% to 96.3%, by tie order)
```

Isotonic has by far the lowest ECE. But look at what it costs:

- **Ties.** Isotonic gives 3,080 answers only 66 different values, and a third of them share a single value. Answers in the same step can no longer be ranked against each other. *Accuracy at 80% coverage* means: keep the 80% of answers with the highest confidence, and see how many are right. The 80% cut falls inside a tied step, so which answers make it is arbitrary. Depending on the order, the result is anywhere from 92.7% to 96.3%. The script reports the average, 94.6%: lower than temperature scaling's 95.1%, which keeps every answer distinct.
- **A flattering ECE.** Few distinct values means few, well-filled bins, and the ECE rewards that. Recall post 4: one constant value scores an ECE of 0. A low ECE doesn't make a confidence better at telling right from wrong. Post 7 measures that directly.
- **Steps that rest on little data.** The top step says 1.000 because all 200 calibration answers in it were right. On the test set, 864 answers got 1.000, and 6 of them were wrong. Isotonic needs more calibration data than a temperature does: it estimates a whole curve, not one number.
- **It only calibrates the chosen answer.** It maps the answer's confidence, not the full distribution over options, so it can't say anything about the runner-up.

```bash
uv run 06-isotonic-regression/plot.py
```

`results/staircase-banking77.png` shows the fitted curve over the calibration answers. `results/reliability-banking77.png` compares the three on the test answers. Its marker sizes show how many answers each bin holds: isotonic's small bins, made of the few answers that fall on the ramps between steps, jump around, while its big ones sit close to the diagonal.
