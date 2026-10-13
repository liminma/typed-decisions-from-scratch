"""How the model's tokenizer splits answers: why the series answers with letters.

    python 02-forcing-a-valid-answer/tokens.py
    python 02-forcing-a-valid-answer/tokens.py "card_arrival" "C-B"     # your own strings
"""

import string
import sys

from server import tokenize

examples = sys.argv[1:] or [
    "A", " A",                           # a letter is one token; with a space before it, a different one
    "76", "151",                         # numbers split into digits
    "card_arrival", "transaction_charged_twice", "reverted_card_payment?",   # intent names: 3-5 tokens
    "C-B", "-B",                         # a code is two tokens: the group letter, then "-" and the intent letter
]
for text in examples:
    pieces = tokenize(text)
    print(f"{text!r:30} {len(pieces)} token{'s' if len(pieces) > 1 else ' '}  {pieces}")

if not sys.argv[1:]:
    # The codes only work if this holds for every letter
    single = all(len(tokenize(s)) == 1 for c in string.ascii_uppercase for s in (c, "-" + c))
    print(f"\nEvery letter A-Z and every '-A' to '-Z' is a single token: {single}")
