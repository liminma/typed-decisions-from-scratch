"""Give the model the list of labels and ask for one. How often is the reply a label?

    python 01-why-typed-decisions/label_list.py              # 200 BANKING77 calibration messages
    python 01-why-typed-decisions/label_list.py --limit 50

The model gets BANKING77's 77 intent names and is told to reply with one of them, and nothing
else. No grammar holds it to that. Each reply is sorted into: an exact label, a label with
extra text around it, or no label at all.
"""

import argparse
import statistics
import time

from data import load
from server import chat

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=200)
args = ap.parse_args()

ds = load("banking77")
intents = [i for group in ds.taxonomy["groups"].values() for i in group]
system = ("You classify messages sent to a digital bank's customer support.\n"
          "Reply with the single best intent from this list, and nothing else:\n" + "\n".join(intents))

kinds = {"an exact label": [], "a label inside other text": [], "no label": []}
correct, ms = 0, []
for row_id, text, gold in ds.split("calib")[:args.limit]:
    start = time.perf_counter()
    reply = chat(system, text, max_tokens=40)["choices"][0]["message"]["content"]
    ms.append((time.perf_counter() - start) * 1000)
    if reply in intents:
        kinds["an exact label"].append((text, reply))
    elif any(i in reply for i in intents):
        kinds["a label inside other text"].append((text, reply))
    else:
        kinds["no label"].append((text, reply))
    correct += reply == gold

n = len(ms)
print(f"{n} BANKING77 messages, the 77 intent names in the prompt, no grammar:")
for kind, replies in kinds.items():
    print(f"  {len(replies):4} replies were {kind}")
print(f"  {correct:4} were the right label ({correct / n:.0%})")
print(f"  median time per request: {statistics.median(ms):.0f} ms\n")
for kind in ("a label inside other text", "no label"):
    for text, reply in kinds[kind][:5]:
        print(f"  {kind}: {text!r} -> {reply!r}")
print("\nEven when every reply is a label, nothing says how sure the model was of it.")
