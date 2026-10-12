"""Build the classification prompt and its grammar from a taxonomy.

The model answers with a two-letter code, `<group letter>-<intent letter>`, for example `C-B`:
group C, intent B within it. Each letter is a single token, so the whole answer is two tokens
("C" and "-B"), and its probabilities can be read straight from the logprobs (post 3).

The system prompt lists every group and intent, so it's the same for every request and the server
keeps it cached; only the user's message is new. The grammar allows only valid codes: after the
group letter, only that group's intent letters.

    python 01-why-typed-decisions/prompt.py banking77     # print the prompt and grammar
"""

import json
import string
import sys
from dataclasses import dataclass

LETTERS = string.ascii_uppercase


@dataclass
class Prompt:
    system: str
    grammar: str                      # GBNF, sent with each request as "grammar"
    codes: dict[str, tuple[str, str]]  # "C-B" -> (group, intent)


def build(taxonomy: dict, shots: dict[str, list[str]] | None = None) -> Prompt:
    """taxonomy: as in datasets/<name>/taxonomy.json. shots: example messages per intent, optional."""
    group_notes = taxonomy.get("group_descriptions", {})
    intent_notes = taxonomy.get("intent_descriptions", {})
    lines, rules, codes = [], [], {}
    for g, (group, intents) in zip(LETTERS, taxonomy["groups"].items()):
        note = group_notes.get(group)
        lines.append(f"{g} = {group}" + (f" ({note})" if note else ""))
        for i, intent in zip(LETTERS, intents):
            codes[f"{g}-{i}"] = (group, intent)
            note = intent_notes.get(intent)
            lines.append(f"  {g}-{i} = {intent}" + (f": {note}" if note else ""))
            if shots and shots.get(intent):
                lines.append("      e.g. " + " | ".join(json.dumps(t, ensure_ascii=False) for t in shots[intent]))
        # This group's letter, a hyphen, then one of its intent letters: ("C-" [A-K])
        rules.append(f'("{g}-" [A-{LETTERS[len(intents) - 1]}])')

    system = (
        f"You are an intent classifier for {taxonomy['about']}.\n"
        "Read the message and choose the single best intent from the list below.\n"
        "Answer with its code only: the group letter, a hyphen, then the intent letter (for example B-C).\n\n"
        "Groups and intents:\n" + "\n".join(lines)
    )
    return Prompt(system, "root ::= " + " | ".join(rules), codes)


if __name__ == "__main__":
    from data import load

    ds = load(sys.argv[1] if len(sys.argv) > 1 else "banking77")
    prompt = build(ds.taxonomy, ds.shots())
    print(prompt.system)
    print("\n--- grammar ---")
    print(prompt.grammar)
