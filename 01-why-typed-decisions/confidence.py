"""Classify a message and read how sure the model was, from the logprobs.

For each token it generates, llama-server can return the most likely candidates for that position
with their log-probabilities (`logprobs: true`, `top_logprobs: N`). Three things to know:

- They are taken BEFORE the grammar and before sampling: the list holds what the model wanted to
  write, including tokens the grammar would never allow ("The", " A", end of text). So we keep
  the candidates that are valid letters at that point and renormalize over them. What's left out
  is the "off-label" mass, worth recording: when it's large, the model wanted to say something else.
- A code has two choices, so there are two distributions: P(group) at the first letter, and
  P(intent | group) at the second, given the group the model picked.
- The confidence of the answer is their product: P(group) x P(intent | group).

    from confidence import classify
    r = classify(prompt, "I was charged twice for the same coffee")
    r["answer"], r["confidence"], r["p_group"], r["p_intent"], r["off_label_group"]
"""

import math

from prompt import Prompt
from server import chat

TOP_LOGPROBS = 30  # enough to cover every letter at both steps (8 groups, up to 15 intents)


def classify(prompt: Prompt, text: str) -> dict:
    params = {"grammar": prompt.grammar, "logprobs": True, "top_logprobs": TOP_LOGPROBS}
    choice = chat(prompt.system, text, max_tokens=2, **params)["choices"][0]
    retried = choice["message"]["content"] not in prompt.codes
    if retried:  # the code was spelled in three tokens ("C", "-", "F"); see post 2
        choice = chat(prompt.system, text, max_tokens=3, **params)["choices"][0]
    answer = choice["message"]["content"]
    probs = read_probs(prompt, answer, choice["logprobs"]["content"])
    group, intent = prompt.codes[answer]
    # The grammar can force a letter the model barely considered: if it isn't among the top
    # candidates at all, its probability is below the last one's, and it counts as 0.
    return {"answer": answer, "group": group, "intent": intent,
            "confidence": probs["p_group"].get(answer[0], 0.0) * probs["p_intent"].get(answer[2], 0.0),
            **probs, "retried": retried}


def read_probs(prompt: Prompt, answer: str, tokens: list[dict]) -> dict:
    """P(group) and P(intent | group) for the code `answer`, from its tokens' top_logprobs.

    A candidate counts for a letter if the text so far plus that candidate starts a valid code
    with that letter: at the first step "C" counts for group C; at the second, "-F" counts for
    intent F (only the chosen group's intents). If the model spelled the code in three tokens,
    the intent is read at the third token instead, where "F" itself is the candidate.
    """
    g = answer[0]
    group_letters = {code[0] for code in prompt.codes}
    intent_letters = {code[2] for code in prompt.codes if code[0] == g}

    def group_of(s):
        return s[0] if s and s[0] in group_letters and (len(s) == 1 or s[1] == "-") else None

    def intent_of(s):
        return s[2] if len(s) >= 3 and s[:2] == g + "-" and s[2] in intent_letters else None

    text, steps = "", {}
    for token in tokens:
        before, text = text, text + token["token"]
        if "group" not in steps:
            steps["group"] = _sum_by_letter(before, token["top_logprobs"], group_of)
        if len(text) >= 3:
            steps["intent"] = _sum_by_letter(before, token["top_logprobs"], intent_of)
            break

    out = {}
    for step, mass in steps.items():
        total = sum(mass.values())
        out[f"p_{step}"] = {letter: p / total for letter, p in sorted(mass.items(), key=lambda kv: -kv[1])}
        out[f"off_label_{step}"] = max(0.0, 1.0 - total)
    return out


def _sum_by_letter(before: str, candidates: list[dict], letter_of) -> dict[str, float]:
    mass = {}
    for c in candidates:
        letter = letter_of(before + c["token"])
        if letter is not None:
            mass[letter] = mass.get(letter, 0.0) + math.exp(c["logprob"])
    return mass
