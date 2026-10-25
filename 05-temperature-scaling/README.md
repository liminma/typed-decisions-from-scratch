# 05 - Fixing the confidence: temperature scaling

Code for Post 5: Fixing the confidence: temperature scaling

Correcting the model's confidence with one number per step, fitted on the calibration answers and checked on the test answers.

| Script | Shows |
|---|---|
| `how_it_works.py` | What a temperature does to a distribution, and why the fit finds the honest value |
| `temperature.py` | `scale()`, `fit_temperature()`, and the fit: temperatures and ECE before and after, on BANKING77 or (with `--dataset clinc150`) CLINC150. Standard library only |
| `plot.py` | Reliability diagram on the test answers, raw and scaled. Needs matplotlib (`uv sync` at the repo root) |

`ece.py` is a copy from [`04-is-the-confidence-honest/`](../04-is-the-confidence-honest/).

These scripts don't call the server. They read the answers that [`03-where-confidence-comes-from/run.py`](../03-where-confidence-comes-from/) saved in `03-where-confidence-comes-from/results/`, for both splits of both datasets. BANKING77's are there if you ran post 3 or 4. CLINC150's are new in this post, so run those first (about 25 minutes):

```bash
python3 03-where-confidence-comes-from/run.py --dataset clinc150 --split calib    # 850 messages, ~4 min
python3 03-where-confidence-comes-from/run.py --dataset clinc150                  # test: 5,500 messages, ~23 min
```

If BANKING77's aren't saved either, run `python3 03-where-confidence-comes-from/run.py --split calib` and `python3 03-where-confidence-comes-from/run.py` too (about 15 minutes).

## 1. What a temperature does

```bash
python3 05-temperature-scaling/how_it_works.py
```

A temperature T turns each probability p into p^(1/T), then renormalizes so they sum to 1 again:

```
   raw                  T = 0.5   T = 0.8   T = 1.0   T = 1.5
   0.60 / 0.30 / 0.10     0.783     0.655     0.600     0.517
   0.80 / 0.15 / 0.05     0.962     0.866     0.800     0.673
   0.95 / 0.04 / 0.01     0.998     0.978     0.950     0.855
```

T below 1 sharpens the distribution, so the top option gains; T above 1 flattens it. The order of the options never changes, so neither does the answer: temperature scaling only moves the confidence.

## 2. How T is chosen

T is chosen to make the right answers as likely as possible, which is the same as making their negative log-likelihood (NLL) as small as possible. That finds the honest value because NLL is a *proper scoring rule*. If answers given confidence q are right 92% of the time, the average loss is smallest when q is exactly 0.92:

```
   q = 0.90   0.2811
   q = 0.92   0.2788   <- lowest
   q = 0.94   0.2820
```

`fit_temperature()` tries 60 values of T from 0.05 to 20 on a log scale, then 25 more within 15% of the best one.

Each step of the code gets its own T:

- **The group step** is fitted on every calibration answer.
- **The intent step** is fitted only on answers whose group was right, because that step only chooses among the chosen group's intents.

The answer's confidence is then the product of the two scaled probabilities, as before.

## 3. The fit on BANKING77

```bash
python3 05-temperature-scaling/temperature.py
```

Your numbers will differ slightly, because they come from your own post 3 answers:

```
Fitted on 770 calibration answers:
  T for the group step:  0.803  (sharpens)
  T for the intent step: 1.138  (flattens)

        answers  accuracy  mean conf. raw  scaled  ECE raw  scaled
calib       770     86.9%           0.824   0.837   0.0545  0.0394
test       3080     86.9%           0.832   0.845   0.0409  0.0301
```

- **The two steps need opposite corrections.** The group probabilities are too flat and get sharpened. The intent probabilities, given the right group, are a little too sharp and get flattened. Overall, the confidence rises, because it was too low (post 4).
- **On the test set, ECE falls from 0.041 to 0.030.** That's the number to report: the temperatures were fitted on the calibration set only.
- **Accuracy doesn't change,** since the answers don't.

```bash
uv run 05-temperature-scaling/plot.py
```

The diagram shows the scaled points closer to the diagonal, but not on it. Each T is a single knob: it sharpens or flattens its step's probabilities everywhere at once. It can't make the model more sure in one range and less sure in another. So the honest claim is "as close to the accuracy as one number allows", not "equal to the accuracy". Post 6 lifts that limit.

## 4. The fit on CLINC150

```bash
python3 05-temperature-scaling/temperature.py --dataset clinc150
```

```
Fitted on 850 calibration answers:
  T for the group step:  0.523  (sharpens)
  T for the intent step: 0.619  (sharpens)

        answers  accuracy  mean conf. raw  scaled  ECE raw  scaled
calib       850     96.2%           0.859   0.959   0.1036  0.0147
test       5500     94.8%           0.860   0.956   0.0885  0.0113
```

The model is far more modest here: right 94.8% of the time with a mean confidence of 0.86. Both steps get sharpened, and ECE on the test set drops from 0.089 to 0.011. The temperatures differ between the two datasets, even in direction, so each task needs its own fit on its own labelled answers.
