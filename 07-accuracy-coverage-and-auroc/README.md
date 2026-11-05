# 07 · Does confidence separate right from wrong?

Code for post 7: honest confidence isn't enough. It also has to rank the answers, so that the wrong ones collect at the bottom, where they can be handed to a person.

These scripts don't call the server. They read the answers saved by [`03-where-confidence-comes-from/run.py`](../03-where-confidence-comes-from/), for both splits of both datasets. If you ran posts 3 and 5, they're already there. Otherwise run these first:

```bash
# BANKING77 (post 3), ~15 min
python3 03-where-confidence-comes-from/run.py --split calib
python3 03-where-confidence-comes-from/run.py
# CLINC150 (post 5)
python3 03-where-confidence-comes-from/run.py --dataset clinc150 --split calib    # ~4 min
python3 03-where-confidence-comes-from/run.py --dataset clinc150                  # ~23 min
```

`ece.py` and `temperature.py` are copies from [`06-isotonic-regression/`](../06-isotonic-regression/). The scripts need scikit-learn and matplotlib (`uv sync`), except `coverage.py`, which uses only the standard library.

| Script | Shows |
|---|---|
| `coverage.py` | Accuracy against coverage: how accuracy rises as you keep only the most confident answers |
| `auroc.py` | AUROC computed three ways, and which changes to the confidence alter it |
| `out_of_scope.py` | CLINC150: spotting requests that match none of the intents |
| `pooling.py` | What a single AUROC over two kinds of traffic is made of |
| `plot.py` | The accuracy–coverage curve, and the ROC curve for out-of-scope requests |

All numbers below are from a Dell Pro Max with GB10.

## 1. Accuracy against coverage

```bash
python3 07-accuracy-coverage-and-auroc/coverage.py
```

Sort the answers from most to least confident. Keep the top share and hand the rest to a person. *Coverage* is the share you keep, and the accuracy of the kept answers is the *selective accuracy*:

```
  coverage  answers kept  confidence at least  accuracy
      100%          3080                0.101     86.9%
       90%          2772                0.497     92.3%
       80%          2464                0.706     94.9%
       60%          1848                0.887     97.5%
       40%          1232                0.950     98.6%
```

Handing the least confident 20% of BANKING77 to a person lifts accuracy on the rest from 86.9% to 94.9%. On CLINC150 (`--dataset clinc150`), it goes from 94.8% to 99.1%. This only works because the wrong answers are, on the whole, the less confident ones.

## 2. AUROC

```bash
uv run 07-accuracy-coverage-and-auroc/auroc.py
```

AUROC measures that ranking in one number: pick a right answer and a wrong one at random. How likely is it that the right one is more confident? 1.0 means always, and 0.5 is a coin flip. The script computes it three ways, which agree:

```
banking77 test: 2675 right and 405 wrong answers, 1,083,375 (right, wrong) pairs

1. Three ways to compute it, on the raw confidence:
   counting pairs:           0.8781
   area under the ROC curve: 0.8781
   sklearn roc_auc_score:    0.8781
```

- **Counting pairs** is the definition: every (right, wrong) pair scores 1 if the right answer is more confident, and ½ for a tie.
- **The area under the ROC curve** gets the same number another way. Lower the threshold step by step, and track the share of right answers above it against the share of wrong answers above it.

AUROC depends only on the *order* of the confidences, not their values:

```
2. Changing the confidence:
   raw                                    AUROC 0.8781   (3079 distinct values)
   raw cubed (same order, other values)   AUROC 0.8781   (3079 distinct values)
   temperature-scaled                     AUROC 0.8874   (3079 distinct values)
   isotonic                               AUROC 0.8707   (66 distinct values)
```

- **Cubing every confidence** changes every value but keeps the order, so AUROC doesn't move.
- **Temperature scaling** can reorder answers slightly: an answer's scaled confidence depends on its whole distribution, not only its top probability, and here the two steps are scaled in opposite directions. Here that helps (0.878 → 0.887); on CLINC150 it hurts a little (0.910 → 0.906).
- **Isotonic** never reorders, but it merges answers into ties, and a tied pair counts only half (0.878 → 0.871).

This is what ECE missed in post 4. A confidence can be perfectly calibrated and rank nothing: a constant value has ECE 0 and AUROC 0.5. Calibration tells you whether the number at a threshold can be trusted. AUROC tells you whether any threshold separates right from wrong.

## 3. Out of scope

```bash
uv run 07-accuracy-coverage-and-auroc/out_of_scope.py
```

CLINC150's test set has 1,000 requests that match none of its 150 intents. They have their own group, `OUT_OF_SCOPE`, so the first step of every answer gives P(OUT_OF_SCOPE):

```
1. P(OUT_OF_SCOPE) as a score:
   median for out-of-scope requests: 0.908
   median for in-scope requests:     0.015
   AUROC: 0.992  (over 4,500,000 (out, in) pairs)

2. Flag a request as out of scope when P(OUT_OF_SCOPE) is at least t:
       t  out-of-scope caught  in-scope flagged by mistake
     0.9                53.0%                        0.07%
     0.5                90.4%                        1.04%
     0.1                98.8%                       10.84%
    0.01                99.8%                       58.76%

3. The model's own answer (group OUT_OF_SCOPE, whatever its probability):
   out-of-scope caught: 92.6%, in-scope flagged by mistake: 1.64%
```

An AUROC of 0.992 says the two kinds of request are almost perfectly separable. It doesn't say where to draw the line: each threshold trades requests caught against requests flagged by mistake. Post 8 chooses one.

## 4. The pooling trap

```bash
uv run 07-accuracy-coverage-and-auroc/pooling.py
```

Pool BANKING77 and CLINC150 and compute a single AUROC. Its (right, wrong) pairs come in four kinds, and the pooled AUROC is their average, weighted by the number of pairs:

```
  right from banking77  vs  wrong from banking77   1,083,375 pairs   AUROC 0.878
  right from banking77  vs  wrong from clinc150      770,400 pairs   AUROC 0.901
  right from clinc150   vs  wrong from banking77   2,110,860 pairs   AUROC 0.882
  right from clinc150   vs  wrong from clinc150    1,501,056 pairs   AUROC 0.910

  weighted average: 0.892
  pooled AUROC:     0.892  (the same number)
```

With the raw confidence, the pooled value lands between the two datasets. Temperature scaling (post 5) changes that, because it makes CLINC150's answers far more confident than BANKING77's:

```bash
uv run 07-accuracy-coverage-and-auroc/pooling.py --scaled
```

```
  right from banking77  vs  wrong from banking77   1,083,375 pairs   AUROC 0.887
  right from banking77  vs  wrong from clinc150      770,400 pairs   AUROC 0.746
  right from clinc150   vs  wrong from banking77   2,110,860 pairs   AUROC 0.973
  right from clinc150   vs  wrong from clinc150    1,501,056 pairs   AUROC 0.906

  weighted average: 0.905
  pooled AUROC:     0.905  (the same number)
```

CLINC150's right answers now outrank BANKING77's wrong ones in 97.3% of pairs, and those are the most numerous pairs. The pooled 0.905 sits well above BANKING77's own 0.887, a ranking that BANKING77's traffic doesn't have. When one kind of traffic is both more accurate and more confident than another, the easy cross pairs can lift the pooled AUROC above every part. So when the traffic is mixed, report each kind's AUROC next to the pooled one.

## 5. Plots

```bash
uv run 07-accuracy-coverage-and-auroc/plot.py
```

`results/coverage-banking77.png` is the accuracy–coverage curve. `results/roc-clinc150-oos.png` is the ROC curve for out-of-scope requests, with the thresholds 0.9, 0.5 and 0.1 marked.
