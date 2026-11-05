"""Out of scope: can the confidence spot requests that match none of the intents?

    uv run 07-accuracy-coverage-and-auroc/out_of_scope.py

CLINC150's test set has 4,500 requests that match one of its 150 intents and 1,000 that match
none. The taxonomy gives those their own group, OUT_OF_SCOPE, so the first step of every answer
already says how likely the model thinks it is that the request is out of scope: P(OUT_OF_SCOPE).
Run 03-where-confidence-comes-from/run.py --dataset clinc150 (both splits) first.
"""

import statistics

from sklearn.metrics import roc_auc_score

from ece import load_answers
from temperature import Letters

letters = Letters("clinc150")
oos_letter = letters.code["oos"][0]
test = load_answers("clinc150", "test")
p_oos = [a["p_group"].get(oos_letter, 0.0) for a in test]
is_oos = [a["gold"] == "oos" for a in test]
n_oos, n_in = sum(is_oos), len(is_oos) - sum(is_oos)

print(f"CLINC150 test: {n_oos} out-of-scope and {n_in} in-scope requests\n")
print("1. P(OUT_OF_SCOPE) as a score:")
print(f"   median for out-of-scope requests: {statistics.median(p for p, o in zip(p_oos, is_oos) if o):.3f}")
print(f"   median for in-scope requests:     {statistics.median(p for p, o in zip(p_oos, is_oos) if not o):.3f}")
print(f"   AUROC: {roc_auc_score(is_oos, p_oos):.3f}  (over {n_oos * n_in:,} (out, in) pairs)\n")

print("2. Flag a request as out of scope when P(OUT_OF_SCOPE) is at least t:")
print(f"   {'t':>5}  {'out-of-scope caught':>19}  {'in-scope flagged by mistake':>27}")
for t in (0.9, 0.5, 0.1, 0.01):
    caught = sum(p >= t for p, o in zip(p_oos, is_oos) if o) / n_oos
    flagged = sum(p >= t for p, o in zip(p_oos, is_oos) if not o) / n_in
    print(f"   {t:5}  {caught:19.1%}  {flagged:27.2%}")
print("   A high AUROC says some threshold separates them well; where to put it is a trade-off (post 8).\n")

# The simplest rule needs no threshold: the answer itself is OUT_OF_SCOPE
chose_oos = [a["answer"][0] == oos_letter for a in test]
print("3. The model's own answer (group OUT_OF_SCOPE, whatever its probability):")
print(f"   out-of-scope caught: {sum(c for c, o in zip(chose_oos, is_oos) if o) / n_oos:.1%}, "
      f"in-scope flagged by mistake: {sum(c for c, o in zip(chose_oos, is_oos) if not o) / n_in:.2%}")
