"""The same messages as a typed decision: a label from the list, a confidence, in one request.

    python 01-why-typed-decisions/typed.py
    python 01-why-typed-decisions/typed.py "My card was declined at the shop"

This is where the series ends up. Post 2 explains the codes and the grammar, post 3 the
confidence, posts 4-6 how to make it honest, and post 8 what to do with it.
"""

import sys
import time

from confidence import classify
from data import load
from prompt import build

ds = load("banking77")
prompt = build(ds.taxonomy, ds.shots())
messages = sys.argv[1:] or [
    "I am still waiting on my card?",
    "I need assistance with understanding which fiat currencies that you support.",
    "How long is the delivery time to the US?",
    "Help me use American Express to add money to the account.",
]
for text in messages:
    start = time.perf_counter()
    r = classify(prompt, text)
    ms = (time.perf_counter() - start) * 1000
    print(f"{text}\n  -> {r['intent']}  (code {r['answer']}, confidence {r['confidence']:.2f}, {ms:.0f} ms)\n")
