# Datasets

The series uses two public intent datasets:

- **[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets):** the running example in posts 1–8. It has 77 intents from a bank's customer support, and many of them overlap, so the model makes mistakes worth calibrating.
- **[CLINC150](https://github.com/clinc/oos-eval):** used in posts 5, 7 and 8. It has 150 intents from a virtual assistant, plus requests that match none of them ("out of scope").

This folder holds no messages, only what the series adds on top of the data: the labels arranged into groups, and which rows go where. The posts' scripts download the data itself into `data/` at the repo root (ignored by git), from pinned commits, and check its SHA-256.

| File | What it is |
|---|---|
| `banking77/taxonomy.json` | 8 groups of BANKING77 intents, and a one-line description of each intent |
| `banking77/splits.json` | Row IDs: 770 calibration, 3,080 test, 5 prompt examples per intent |
| `clinc150/taxonomy.json` | CLINC150's 10 official domains as groups, plus `OUT_OF_SCOPE` |
| `clinc150/splits.json` | Row IDs: 850 calibration, 5,500 test, 5 prompt examples per intent |
| `make_splits.py` | Rebuilds the split files from the data, or makes them for your own data |

## Licences and credit

- **BANKING77** by PolyAI, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", NLP4ConvAI workshop, 2020.
- **CLINC150** by Clinc, licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", EMNLP 2019.

The taxonomies and splits here are derived from these datasets. If you reuse them, credit the datasets as above.

## The files

**Row IDs** name a row of the original files by its position, starting at 0 with the first row after the header. `b77-test-0` is the first message in BANKING77's `test.csv`, and `b77-test-11` is the 12th. `clinc-validation-40` is the 41st message of CLINC150's validation split in the `plus` version (`data_oos_plus.json`), with the in-scope rows first and the out-of-scope ones after them.

**`taxonomy.json`**:

```json
{
  "about": "customer messages sent to a digital bank's customer support",
  "groups": {
    "CARD_SETUP": ["card_arrival", "card_delivery_estimate", "..."],
    "...": []
  },
  "group_descriptions": {"OUT_OF_SCOPE": "the request does not match any intent listed above"},
  "intent_descriptions": {"card_arrival": "a card that was sent has not arrived yet; tracking its delivery"}
}
```

`about` says what the messages are; the prompt opens with "You are an intent classifier for <about>." `groups` lists every intent exactly once, in the order the prompt shows them. The order matters: the model answers with letters, so `CARD_SETUP` is `A` and its first intent is `A-A` (post 2 explains why letters). Both description fields are optional.

**`splits.json`**:

```json
{"seed": 13, "calib": ["b77-train-0", "..."], "test": ["b77-test-0", "..."], "shots": {"card_arrival": ["b77-train-24", "..."]}}
```

- `calib`: rows for fitting calibration and choosing thresholds (posts 4–8).
- `test`: rows for reporting results. Nothing is ever fitted or chosen on them.
- `shots`: example messages the prompt shows under each intent. They come from rows in neither split.

## How the series' files were made

**BANKING77 groups.** BANKING77 has no groups of its own, and 77 intents are too many to list behind single letters, so the series sorts them into 8 groups of at most 14. The groups were drafted, then reviewed against the training rows. The review moved six intents, so that intents easily confused with each other end up in the same group, where the model chooses between them side by side. `get_physical_card` (in practice, about the card's PIN) and `declined_card_payment` moved to the card problems group, renamed `CARD_PROBLEMS_AND_PIN`. `apple_pay_or_google_pay`, `supported_cards_and_currencies` and `transfer_into_account` moved to `TOP_UP`, and `country_support` to `ACCOUNT_AND_IDENTITY`.

**BANKING77 descriptions.** One line per intent, written by reading training messages that aren't in `calib` (see **Splits** below). Some describe how the dataset actually uses an intent rather than what its name says. For example, `get_physical_card` is "the PIN for a physical card: where to find it or it has not arrived", and `apple_pay_or_google_pay` is "using Apple Pay or Google Pay, mostly to top up".

**CLINC150 groups.** The 10 domains that come with the dataset ([`domains.json`](https://github.com/clinc/oos-eval/blob/master/data/domains.json)), in that file's order with intents sorted, plus `OUT_OF_SCOPE` with the single intent `oos`. There are no intent descriptions; CLINC150's intent names are clear enough on their own.

**Splits**, all drawn with seed 13:

- **BANKING77.** `calib` has 10 rows per intent from the official training set. `test` is the official test set.
- **CLINC150.** `calib` has 5 rows per in-scope intent from the official validation set, plus its 100 out-of-scope rows. `test` is the official test set: 4,500 in scope and 1,000 out of scope.
- **Shots.** 5 per intent, from all rows outside `calib` and `test`.

To check the shipped files against the data:

```bash
python3 datasets/make_splits.py banking77
python3 datasets/make_splits.py clinc150
```

## Making your own

To run the series on your own labelled messages:

1. **Put the data in a CSV** with a `text` and a `label` column, one message per row.

2. **Write a `taxonomy.json`.**
   - With 26 labels or fewer, use a single group.
   - With more, sort them into groups. There can be at most 26 groups and at most 26 intents per group, since each is one letter. Smaller groups are easier for the model; the series keeps them to about a dozen.
   - Put intents that are easy to confuse in the same group.
   - If you have an existing grouping (CLINC150's domains, a product's menu), start from that.

3. **Make the splits:**

   ```bash
   python3 datasets/make_splits.py csv my_data.csv datasets/my_data/taxonomy.json \
       --calib 10 --test 60 --shots 5 > datasets/my_data/splits.json
   ```

   This draws `calib` and `test` per label, without overlap, and takes the shots from the rows left over. Your IDs are `row-0`, `row-1`, … by position in the CSV, so don't reorder the file afterwards.

   - **Calibration:** posts 5 and 6 need a few hundred `calib` rows in total to fit anything.
   - **Test:** needs enough rows per label for the numbers to mean something.
   - **Small labels:** a label with fewer rows than you ask for gives all it has to `calib`, and then nothing is left for `test`.

4. **Optionally, add `intent_descriptions`.** They help most where label names are vague or the dataset uses a label in an unexpected way. Write them only from rows that are in neither `calib` nor `test`, and read the model's mistakes on `calib`, never on `test`. Otherwise the test numbers will flatter you.
