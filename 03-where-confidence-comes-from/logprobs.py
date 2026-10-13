"""From logprobs to a confidence, step by step, for one message.

    python 03-where-confidence-comes-from/logprobs.py
    python 03-where-confidence-comes-from/logprobs.py "Where is my new card?"
"""

import math
import sys

from confidence import TOP_LOGPROBS, read_probs
from data import load
from prompt import build
from server import chat

text = sys.argv[1] if len(sys.argv) > 1 else "I was charged twice for the same coffee, can I get one refunded?"
ds = load("banking77")
prompt = build(ds.taxonomy, ds.shots())
groups = {code[0]: group for code, (group, _) in prompt.codes.items()}

params = {"grammar": prompt.grammar, "logprobs": True, "top_logprobs": TOP_LOGPROBS}
choice = chat(prompt.system, text, max_tokens=2, **params)["choices"][0]
if choice["message"]["content"] not in prompt.codes:  # spelled in three tokens; see post 2
    choice = chat(prompt.system, text, max_tokens=3, **params)["choices"][0]
answer = choice["message"]["content"]
tokens = choice["logprobs"]["content"]
print(f"Message: {text}")
print(f"Answer:  {answer} = {prompt.codes[answer][1]} (in {prompt.codes[answer][0]})\n")

# 1. What the server returns: for each generated token, the top candidates and their logprobs.
#    Some aren't letters at all: they show what the model would have written without the grammar.
for n, token in enumerate(tokens, 1):
    print(f"Token {n}: {token['token']!r}. Top candidates, before the grammar:")
    for c in token["top_logprobs"][:10]:
        print(f"  {c['token']!r:8} logprob {c['logprob']:8.3f}   p = {math.exp(c['logprob']):.4f}")
    print()

# 2. Keep the valid letters, renormalize, and record what was left out.
probs = read_probs(prompt, answer, tokens)
g, i = answer[0], answer[2]
print(f"P(group), over the {len(groups)} group letters ({probs['off_label_group']:.2%} of the probability was on other tokens):")
for letter, p in list(probs["p_group"].items())[:5]:
    print(f"  {letter} {groups[letter]:34} {p:.4f}")
print(f"\nP(intent | group {g}), over that group's letters ({probs['off_label_intent']:.2%} off-label):")
for letter, p in list(probs["p_intent"].items())[:5]:
    print(f"  {g}-{letter} {prompt.codes[g + '-' + letter][1]:34} {p:.4f}")

# 3. The answer's confidence: both choices had to be right.
pg, pi = probs["p_group"].get(g, 0.0), probs["p_intent"].get(i, 0.0)  # 0 if not among the top candidates
print(f"\nConfidence in {answer}: P(group {g}) x P(intent {i} | group {g}) = {pg:.4f} x {pi:.4f} = {pg * pi:.4f}")
