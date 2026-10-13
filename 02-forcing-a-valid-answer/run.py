"""Classify a whole split and count valid answers, with or without the grammar.

    python 02-forcing-a-valid-answer/run.py                         # BANKING77 test, with the grammar (~16 min)
    python 02-forcing-a-valid-answer/run.py --split calib --limit 200 --no-grammar
    python 02-forcing-a-valid-answer/run.py --dataset clinc150

Each answer is saved as a line of results/<dataset>-<split>-<grammar|no-grammar>.jsonl in this
folder, so the counts can be checked afterwards.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from classify import classify
from data import load
from prompt import build

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="banking77", choices=("banking77", "clinc150"))
ap.add_argument("--split", default="test", choices=("calib", "test"))
ap.add_argument("--limit", type=int, help="only the first N rows")
ap.add_argument("--no-grammar", dest="grammar", action="store_false")
args = ap.parse_args()

ds = load(args.dataset)
prompt = build(ds.taxonomy, ds.shots())
rows = ds.split(args.split)[:args.limit]
out = Path(__file__).parent / "results" / f"{args.dataset}-{args.split}-{'grammar' if args.grammar else 'no-grammar'}.jsonl"
out.parent.mkdir(exist_ok=True)

counts = Counter()
start = time.time()
with out.open("w") as f:
    for n, (row_id, text, intent) in enumerate(rows, 1):
        r = classify(prompt, text, args.grammar)
        r |= {"id": row_id, "intent": intent,
              "predicted": prompt.codes[r["answer"]][1] if r["valid"] else None}
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts["valid" if r["valid"] else "invalid"] += 1
        counts["retried"] += r["retried"]
        counts["correct"] += r["predicted"] == intent
        if n % 100 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

n = len(rows)
print(f"{args.dataset} {args.split}, {'with' if args.grammar else 'without'} the grammar: {n} messages in {time.time() - start:.0f} s")
print(f"  valid codes:  {counts['valid']} ({counts['valid'] / n:.1%})")
print(f"  retried:      {counts['retried']}")
print(f"  correct:      {counts['correct']} ({counts['correct'] / n:.1%})")
print(f"  answers in {out}")
