"""Temperature scaling: one number per step that makes the confidence honest.

    python3 05-temperature-scaling/temperature.py                 # BANKING77: fit on calib, report on test
    python3 05-temperature-scaling/temperature.py --dataset clinc150

Reads the answers saved by 03-where-confidence-comes-from/run.py. Standard library only.

A temperature T rescales a distribution: p_i becomes p_i^(1/T), renormalized so it sums to 1.
T < 1 sharpens it (the top option gains), T > 1 flattens it. The order of the options never
changes, so neither does the answer; only the confidence moves.

Each step of the code gets its own T: one for P(group), one for P(intent | group). T is chosen
on the calibration answers, as the value that makes the right answers most likely (the lowest
negative log-likelihood, NLL), and then applied unchanged to the test answers.
"""

import argparse
import json
import math
import string
from pathlib import Path

from ece import ece, load_answers

EPS = 1e-6  # a letter missing from top_logprobs gets this probability rather than 0


def scale(probs: dict[str, float], letters: list[str], t: float) -> dict[str, float]:
    """probs over `letters`, rescaled by temperature t and renormalized."""
    logs = {k: math.log(max(probs.get(k, 0.0), EPS)) / t for k in letters}
    top = max(logs.values())  # subtract the largest before exp(), so nothing overflows
    weights = {k: math.exp(v - top) for k, v in logs.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def nll(rows: list[tuple[dict, list[str], str]], t: float) -> float:
    """rows: (probabilities, the allowed letters, the right letter). Lower is better."""
    return -sum(math.log(scale(p, letters, t)[gold]) for p, letters, gold in rows)


def fit_temperature(rows: list[tuple[dict, list[str], str]]) -> float:
    """The T with the lowest NLL: 60 values from 0.05 to 20 on a log scale, then 25 values
    within 15% of the best one."""
    def log_grid(lo, hi, n):
        return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1)) for i in range(n)]

    best = min(log_grid(0.05, 20, 60), key=lambda t: nll(rows, t))
    return min(log_grid(best / 1.15, best * 1.15, 25), key=lambda t: nll(rows, t))


class Letters:
    """Which letters each step allows, and the right code for each intent (from the taxonomy)."""

    def __init__(self, dataset: str):
        path = Path(__file__).resolve().parents[1] / "datasets" / dataset / "taxonomy.json"
        groups = json.loads(path.read_text())["groups"]
        L = string.ascii_uppercase
        self.groups = list(L[:len(groups)])
        self.intents = {L[g]: list(L[:len(intents)]) for g, intents in enumerate(groups.values())}
        self.code = {intent: f"{L[g]}-{L[i]}" for g, intents in enumerate(groups.values())
                     for i, intent in enumerate(intents)}


def fit(answers: list[dict], letters: Letters) -> tuple[float, float]:
    """T for the group step, from every answer; T for the intent step, only from answers whose
    group was right, since the intent step only chooses among the chosen group's intents."""
    group_rows, intent_rows = [], []
    for a in answers:
        gold = letters.code[a["gold"]]
        group_rows.append((a["p_group"], letters.groups, gold[0]))
        if a["answer"][0] == gold[0]:
            intent_rows.append((a["p_intent"], letters.intents[gold[0]], gold[2]))
    return fit_temperature(group_rows), fit_temperature(intent_rows)


def scaled_confidence(a: dict, letters: Letters, t_group: float, t_intent: float) -> float:
    g, i = a["answer"][0], a["answer"][2]
    return scale(a["p_group"], letters.groups, t_group)[g] * scale(a["p_intent"], letters.intents[g], t_intent)[i]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77")
    args = ap.parse_args()

    letters = Letters(args.dataset)
    calib, test = load_answers(args.dataset, "calib"), load_answers(args.dataset, "test")
    t_group, t_intent = fit(calib, letters)
    print(f"Fitted on {len(calib)} calibration answers:")
    print(f"  T for the group step:  {t_group:.3f}  ({'sharpens' if t_group < 1 else 'flattens'})")
    print(f"  T for the intent step: {t_intent:.3f}  ({'sharpens' if t_intent < 1 else 'flattens'})\n")

    print(f"{'':6} {'answers':>8} {'accuracy':>9} {'mean conf. raw':>15} {'scaled':>7} {'ECE raw':>8} {'scaled':>7}")
    for name, answers in (("calib", calib), ("test", test)):
        correct = [a["correct"] for a in answers]
        raw = [a["confidence"] for a in answers]
        scaled = [scaled_confidence(a, letters, t_group, t_intent) for a in answers]
        print(f"{name:6} {len(answers):8} {sum(correct) / len(correct):9.1%} {sum(raw) / len(raw):15.3f} "
              f"{sum(scaled) / len(scaled):7.3f} {ece(raw, correct):8.4f} {ece(scaled, correct):7.4f}")
    print("\nThe test row is the one to report: nothing was fitted on it.")
