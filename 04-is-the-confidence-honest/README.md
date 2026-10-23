# 04 - Is the confidence honest?

Code for Post 4: Is the confidence honest?

Checking whether the model's confidence means what it says. When it answers with 0.8, is it right 80% of the time?

| Script | Shows |
|---|---|
| `ece.py` | The table behind the reliability diagram and the ECE. Standard library only |
| `plot.py` | The reliability diagram saved as a PNG in `results/`. Needs matplotlib (`uv sync` at the repo root) |

These scripts don't call the server. They read the answers that [`03-where-confidence-comes-from/run.py`](../03-where-confidence-comes-from/) saved in `03-where-confidence-comes-from/results/`. If you ran it for both splits in post 3, you're ready. If not, run it first (about 15 minutes):

```bash
python3 03-where-confidence-comes-from/run.py --split calib    # 770 messages, ~3 min
python3 03-where-confidence-comes-from/run.py                  # test: 3,080 messages, ~12 min
```

## 1. Bins, gaps and ECE

```bash
python3 04-is-the-confidence-honest/ece.py
```

`ece.py` sorts the answers that post 3 saved into 15 bins by their confidence: 0–0.067, 0.067–0.133, and so on. In each bin, it compares the mean confidence with the share of answers that were right. For the BANKING77 test set, with some rows left out (your numbers will differ slightly, because they come from your own post 3 answers):

```
banking77 test: 3080 answers, accuracy 86.9%, mean confidence 83.2%

          bin  answers  confidence  accuracy     gap  adds to ECE
0.067-0.133        5       0.112     0.000  -0.112       0.0002
0.133-0.200       16       0.169     0.188  +0.019       0.0001
...
0.467-0.533       86       0.501     0.605  +0.104       0.0029
0.533-0.600      107       0.565     0.692  +0.126       0.0044
...
0.867-0.933      537       0.903     0.948  +0.044       0.0078
0.933-1.000     1451       0.967     0.982  +0.016       0.0073

ECE, 15 bins: 0.0409
```

- **`ece.py` lists only 14 of the 15 bins.** The first, 0–0.067, is empty: no answer had a confidence that low.
- **The gap is accuracy minus confidence, and it's mostly positive.** From 0.4 upwards, the answers in every bin were right more often than their confidence said. Answers given with about 0.5 were right 60% of the time. The model is *under-confident*.
- **ECE (expected calibration error)** adds up the gaps without their sign, each weighted by its bin's share of the answers. Here it's 0.041: on average, the confidence is about 4 points away from the accuracy of answers like it.
- **The top bin holds 1,451 of the 3,080 answers,** nearly half, but its gap is small. More than half of the ECE comes from the bins between 0.4 and 0.867, where the gaps are larger. The lowest bins hold 5 and 16 answers, too few to say much.

## 2. The reliability diagram

```bash
uv run 04-is-the-confidence-honest/plot.py
```

The same table as a picture, saved as `results/reliability-banking77-test.png` in this folder. Each point is a bin: its mean confidence on the horizontal axis, its accuracy on the vertical axis. On the dashed diagonal, confidence equals accuracy; points above it are under-confident, points below it over-confident. The light vertical lines are the gaps, and the bars underneath show how many answers each bin holds.

## 3. What ECE doesn't tell you

`ece.py` (section 1) ends with two checks:

```
The same answers with other bin counts:
   5 bins: ECE 0.0402
  10 bins: ECE 0.0404
  15 bins: ECE 0.0409
  20 bins: ECE 0.0410
  30 bins: ECE 0.0411

Every answer given the same confidence, 0.869 (the accuracy): ECE 0.0000
```

- **The number depends on the bins.** On the 3,080 test answers it barely moves. On the 770 calibration answers it ranges from 0.045 with 5 bins to 0.060 with 30: the fewer answers in a bin, the bigger its chance gap, and ECE adds chance gaps up with the real ones. When you report an ECE, say how many bins you used, and compare only numbers that used the same ones. The series always uses 15 bins of equal width.
- **A perfect ECE can be useless.** Imagine giving every answer the same confidence, equal to the overall accuracy (the last line of the output). Every answer then lands in one bin whose confidence and accuracy match, so ECE is 0. But a confidence that's the same for every answer can't tell a right answer from a wrong one, so it can't decide which answers to hand to a human agent. Post 7 measures how well the confidence does that.

## 4. Calibration set and test set

```bash
python3 04-is-the-confidence-honest/ece.py --split calib
uv run 04-is-the-confidence-honest/plot.py --split calib
```

The same table and diagram for the calibration set: 770 messages from BANKING77's training data, 10 per intent. The diagram is saved as `results/reliability-banking77-calib.png`.

Why two sets? Posts 5 and 6 fix the confidence by fitting a correction. They fit it on the **calibration set** and measure it on the **test set** (3,080 messages), which nothing is ever fitted on. A correction measured on the answers it was fitted to looks better than it will on new messages, because it has adapted to their quirks.

The calibration set shows the same pattern as the test set: under-confident in most bins, with an ECE of 0.055. Its diagram is noisier, because 770 answers spread over 15 bins leave some bins with only a handful. The "other bin counts" lines at the end of the first command's output are where section 3's range of 0.045 to 0.060 comes from.
