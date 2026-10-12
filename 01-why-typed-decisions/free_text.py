"""Ask a model what a customer wants in plain words, and look at what comes back.

    python 01-why-typed-decisions/free_text.py
    python 01-why-typed-decisions/free_text.py "My card was declined at the shop"

The reply is prose: fine for a person, but there is nothing in it a program can branch on.
"""

import sys

from server import chat

SYSTEM = "You help a bank's customer support team. Say what the customer wants."
messages = sys.argv[1:] or [
    "I am still waiting on my card?",
    "I need assistance with understanding which fiat currencies that you support.",
    "How long is the delivery time to the US?",
]
for text in messages:
    r = chat(SYSTEM, text, max_tokens=120)
    reply = r["choices"][0]["message"]["content"]
    t = r["timings"]
    print(f"Customer: {text}")
    print(f"Model:    {reply}")
    print(f"          ({t['predicted_n']} tokens, {t['prompt_ms'] + t['predicted_ms']:.0f} ms on the server)\n")
