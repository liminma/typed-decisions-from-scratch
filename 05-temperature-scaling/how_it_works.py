"""How temperature scaling works, on small examples and then on the real calibration answers.

    python3 05-temperature-scaling/how_it_works.py
"""

import math

from ece import load_answers
from temperature import Letters, nll, scale

# 1. What a temperature does to one distribution. The order stays; the values move.
print("1. One distribution at different temperatures (top option's probability):")
dists = [[0.60, 0.30, 0.10], [0.80, 0.15, 0.05], [0.95, 0.04, 0.01], [0.80, 0.20]]
temps = (0.5, 0.8, 1.0, 1.5)
print(f"   {'raw':18}" + "".join(f"{'T = ' + str(t):>10}" for t in temps))
for d in dists:
    probs = {str(i): p for i, p in enumerate(d)}
    row = "".join(f"{scale(probs, list(probs), t)['0']:10.3f}" for t in temps)
    print(f"   {' / '.join(f'{p:.2f}' for p in d):18}{row}")
print("   T below 1 sharpens: the top option gains. T above 1 flattens. T = 1 changes nothing.\n")

# 2. Why NLL finds the honest value. Suppose answers rated q are right 92% of the time.
#    The average loss -[0.92 log q + 0.08 log(1 - q)] is smallest at q = 0.92: stating the
#    true rate is the only way to minimize it (a "proper scoring rule").
print("2. Answers that are right 92% of the time, given confidence q. Average NLL:")
for q in (0.80, 0.85, 0.90, 0.92, 0.94, 0.97, 0.99):
    loss = -(0.92 * math.log(q) + 0.08 * math.log(1 - q))
    print(f"   q = {q:.2f}   {loss:.4f}" + ("   <- lowest" if q == 0.92 else ""))
print()

# 3. The same search on the real calibration answers, for the group step.
letters = Letters("banking77")
rows = [(a["p_group"], letters.groups, letters.code[a["gold"]][0]) for a in load_answers("banking77", "calib")]
print(f"3. NLL of the right group, over the {len(rows)} BANKING77 calibration answers:")
losses = {t: nll(rows, t) for t in (0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)}
for t, loss in losses.items():
    print(f"   T = {t:.1f}   {loss:8.2f}")
best = min(losses, key=losses.get)
print(f"   The lowest of these is at T = {best}: the raw group probabilities are "
      f"{'too flat, so the fit sharpens them' if best < 1 else 'too sharp, so the fit flattens them'}.")
