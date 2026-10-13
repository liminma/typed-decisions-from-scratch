"""Classify a split with confidences, and see whether high confidence goes with right answers.

    python 03-where-confidence-comes-from/run.py                     # BANKING77 test (~12 min)
    python 03-where-confidence-comes-from/run.py --split calib       # 770 rows (~3 min)
    python 03-where-confidence-comes-from/run.py --dataset clinc150

Each answer, with its probabilities, is saved as a line of results/<dataset>-<split>.jsonl in this
folder. Posts 4-8 start from these files.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from confidence import classify
from data import load
from prompt import build

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="banking77", choices=("banking77", "clinc150"))
ap.add_argument("--split", default="test", choices=("calib", "test"))
ap.add_argument("--limit", type=int, help="only the first N rows")
args = ap.parse_args()

ds = load(args.dataset)
prompt = build(ds.taxonomy, ds.shots())
rows = ds.split(args.split)[:args.limit]
out = Path(__file__).parent / "results" / f"{args.dataset}-{args.split}.jsonl"
out.parent.mkdir(exist_ok=True)

results = []
start = time.time()
with out.open("w") as f:
    for n, (row_id, text, intent) in enumerate(rows, 1):
        r = {"id": row_id, "text": text, "gold": intent, **classify(prompt, text)}
        r["correct"] = r["intent"] == intent
        results.append(r)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if n % 100 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

right = [r["confidence"] for r in results if r["correct"]]
wrong = [r["confidence"] for r in results if not r["correct"]]
print(f"\n{args.dataset} {args.split}: {len(results)} messages in {time.time() - start:.0f} s")
print(f"  correct: {len(right)} ({len(right) / len(results):.1%})")
print(f"  median confidence when right: {statistics.median(right):.3f}")
if wrong:
    print(f"  median confidence when wrong: {statistics.median(wrong):.3f}")

for step in ("group", "intent"):
    off = sorted(r[f"off_label_{step}"] for r in results)
    print(f"  off-label mass at the {step} step: median {statistics.median(off):.2%}, "
          f"95th percentile {off[int(0.95 * (len(off) - 1))]:.2%}, max {off[-1]:.2%}")

# A first look at post 4's question: when the model says 0.9, is it right 90% of the time?
print("\n  confidence    messages   correct")
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
    band = [r for r in results if lo <= r["confidence"] < hi]
    if band:
        acc = sum(r["correct"] for r in band) / len(band)
        print(f"  {lo:.2f}-{min(hi, 1):.2f}   {len(band):9}   {acc:7.1%}")
print(f"\n  answers in {out}")
