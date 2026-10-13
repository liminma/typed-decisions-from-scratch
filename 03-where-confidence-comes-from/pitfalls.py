"""Four ways to get the wrong probabilities from llama-server, each shown on one message.

    python 03-where-confidence-comes-from/pitfalls.py
"""

import math

from confidence import read_probs
from data import load
from prompt import build
from server import chat

ds = load("banking77")
prompt = build(ds.taxonomy, ds.shots())
text = "I was charged twice for the same coffee, can I get one refunded?"


def first_step(**params) -> list[tuple[str, float]]:
    choice = chat(prompt.system, text, logprobs=True, **params)["choices"][0]
    first = choice["logprobs"]["content"][0]
    return [(c["token"], round(math.exp(c["logprob"]) if "logprob" in c else c["prob"], 4))
            for c in first.get("top_logprobs") or first.get("top_probs")]


print("1. The grammar doesn't change the logprobs: they're taken before it, so renormalize yourself.")
with_grammar = first_step(grammar=prompt.grammar, max_tokens=2, top_logprobs=8)
without = first_step(max_tokens=2, top_logprobs=8)
print(f"   with the grammar:    {with_grammar}")
print(f"   without the grammar: {without}")
print(f"   identical: {with_grammar == without}\n")

print("2. post_sampling_probs gives the probabilities after sampling. At temperature 0 that's 1.0 for")
print("   the chosen token and nothing else.")
print(f"   {first_step(grammar=prompt.grammar, max_tokens=2, top_logprobs=8, post_sampling_probs=True)}\n")

print("3. Too few top_logprobs, and some letters never appear: they get probability 0 instead of a")
print("   small number, and the answer's share is overstated.")
for n in (3, 30):
    choice = chat(prompt.system, text, grammar=prompt.grammar, max_tokens=2, logprobs=True, top_logprobs=n)["choices"][0]
    answer = choice["message"]["content"]
    probs = read_probs(prompt, answer, choice["logprobs"]["content"])
    print(f"   top_logprobs={n:<3} group letters seen: {len(probs['p_group'])} of 8, "
          f"P(group {answer[0]}) = {probs['p_group'][answer[0]]:.4f}")
print()

print("4. Thinking on: the first token starts the model's reasoning, not the answer, so the first")
print("   step's probabilities say nothing about the label. 00-setup starts the server with")
print("   --reasoning off; a request can still turn it back on:")
print(f"   {first_step(max_tokens=2, top_logprobs=5, chat_template_kwargs={'enable_thinking': True})}")
