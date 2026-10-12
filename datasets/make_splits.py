"""Download the series' datasets and rebuild their split files, or make split files for your own data.

A split file lists row IDs only, never text:
    {"seed": 13, "calib": [ids], "test": [ids], "shots": {intent: [ids]}}
calib is for fitting calibration and thresholds, test is for reporting, and shots are the
example messages the prompt shows for each intent (taken from rows in neither split).

    python datasets/make_splits.py banking77     # rebuild and compare with datasets/banking77/splits.json
    python datasets/make_splits.py clinc150
    python datasets/make_splits.py csv my.csv datasets/my/taxonomy.json --calib 10 --test 60 > datasets/my/splits.json

Standard library only. Downloads go to data/ at the repo root (ignored by git).
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SEED = 13

# Pinned sources: the commit and SHA-256 of every file the series downloads
BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/{}"
BANKING77_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
CLINC150 = "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_oos_plus.json"
CLINC150_SHA256 = "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f"


def fetch(url: str, sha256: str, dest: Path) -> Path:
    """Download once, and check the file is the one the series used."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if got != sha256:
        sys.exit(f"{dest}: SHA-256 {got}, expected {sha256}. Delete it and run again.")
    return dest


def load_banking77() -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    """Rows as {id: (text, intent)}, and the IDs of each official split. ID = b77-<split>-<row>."""
    rows, split_ids = {}, {}
    for split in ("train", "test"):
        path = fetch(BANKING77.format(f"{split}.csv"), BANKING77_SHA256[f"{split}.csv"], DATA / "banking77" / f"{split}.csv")
        with path.open(newline="") as f:
            split_ids[split] = []
            for i, row in enumerate(csv.DictReader(f)):
                rows[f"b77-{split}-{i}"] = (row["text"], row["category"])
                split_ids[split].append(f"b77-{split}-{i}")
    return rows, split_ids


def load_clinc150() -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    """The "plus" version (250 out-of-scope training rows). ID = clinc-<split>-<row>, where each
    split lists the in-scope rows first, then the out-of-scope ones (intent "oos")."""
    raw = json.loads(fetch(CLINC150, CLINC150_SHA256, DATA / "clinc150" / "data_oos_plus.json").read_text())
    rows, split_ids = {}, {}
    for split, parts in (("train", ("train", "oos_train")), ("validation", ("val", "oos_val")), ("test", ("test", "oos_test"))):
        split_ids[split] = []
        for i, (text, intent) in enumerate(r for p in parts for r in raw[p]):
            rows[f"clinc-{split}-{i}"] = (text, intent)
            split_ids[split].append(f"clinc-{split}-{i}")
    return rows, split_ids


def by_label(rows: dict[str, tuple[str, str]], ids) -> dict[str, list[str]]:
    out = defaultdict(list)
    for i in ids:
        out[rows[i][1]].append(i)
    return out


def stratified(ids_by_label: dict[str, list[str]], per_label: int, rng: random.Random) -> list[str]:
    """per_label random IDs from each label (all of them if it has fewer)."""
    out = []
    for label in sorted(ids_by_label):
        ids = sorted(ids_by_label[label])
        out += rng.sample(ids, min(per_label, len(ids)))
    return sorted(out)


def pick_shots(rows: dict[str, tuple[str, str]], taxonomy: dict, used: set[str], per_intent: int) -> dict[str, list[str]]:
    """per_intent example rows for each intent, in the taxonomy's order, never from calib or test."""
    pool = defaultdict(list)
    for i, (_, intent) in rows.items():
        if i not in used:
            pool[intent].append(i)
    rng = random.Random(SEED)
    return {intent: rng.sample(sorted(pool[intent]), min(per_intent, len(pool[intent])))
            for intents in taxonomy["groups"].values() for intent in intents}


def banking77() -> dict:
    """calib: 10 per intent from the official train split; test: the official test split."""
    rows, split_ids = load_banking77()
    rng = random.Random(SEED)
    calib = stratified(by_label(rows, split_ids["train"]), 10, rng)
    return finish(rows, calib, split_ids["test"], HERE / "banking77" / "taxonomy.json")


def clinc150() -> dict:
    """calib: 5 per in-scope intent plus all 100 out-of-scope rows, from the official validation
    split; test: the official test split (4,500 in scope, 1,000 out of scope)."""
    rows, split_ids = load_clinc150()
    rng = random.Random(SEED)
    ids = by_label(rows, split_ids["validation"])
    oos = ids.pop("oos")
    calib = sorted(stratified(ids, 5, rng) + oos)
    return finish(rows, calib, split_ids["test"], HERE / "clinc150" / "taxonomy.json")


def from_csv(path: Path, taxonomy_path: Path, calib_per: int, test_per: int, shots_per: int) -> dict:
    """Your own data: a CSV with `text` and `label` columns. ID = row-<row>. calib and test are
    drawn per label, without overlap; shots come from the rows left over."""
    with path.open(newline="") as f:
        rows = {f"row-{i}": (r["text"], r["label"]) for i, r in enumerate(csv.DictReader(f))}
    rng = random.Random(SEED)
    ids = by_label(rows, rows)
    calib = stratified(ids, calib_per, rng)
    taken = set(calib)
    test = stratified({k: [i for i in v if i not in taken] for k, v in ids.items()}, test_per, rng)
    return finish(rows, calib, test, taxonomy_path, shots_per)


def finish(rows, calib, test, taxonomy_path: Path, shots_per: int = 5) -> dict:
    taxonomy = json.loads(taxonomy_path.read_text())
    labels = {intent for _, intent in rows.values()}
    listed = {i for intents in taxonomy["groups"].values() for i in intents}
    assert labels == listed, f"labels not in {taxonomy_path.name}: {labels - listed}; listed but unused: {listed - labels}"
    assert not set(calib) & set(test), "calib and test overlap"
    return {"seed": SEED, "calib": calib, "test": test,
            "shots": pick_shots(rows, taxonomy, set(calib) | set(test), shots_per)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="dataset", required=True)
    for name in ("banking77", "clinc150"):
        p = sub.add_parser(name)
        p.add_argument("--write", action="store_true", help="overwrite the shipped splits.json")
    p = sub.add_parser("csv", help="your own data")
    p.add_argument("path", type=Path)
    p.add_argument("taxonomy", type=Path)
    p.add_argument("--calib", type=int, default=10, help="calibration rows per label")
    p.add_argument("--test", type=int, default=60, help="test rows per label")
    p.add_argument("--shots", type=int, default=5, help="prompt examples per label")
    args = ap.parse_args()

    if args.dataset == "csv":
        print(json.dumps(from_csv(args.path, args.taxonomy, args.calib, args.test, args.shots), indent=1))
        return
    splits = {"banking77": banking77, "clinc150": clinc150}[args.dataset]()
    path = HERE / args.dataset / "splits.json"
    if args.write:
        path.write_text(json.dumps(splits, indent=1) + "\n")
        print(f"wrote {path}")
    elif json.loads(path.read_text()) == splits:
        print(f"{path.relative_to(HERE.parent)}: identical ({len(splits['calib'])} calib, {len(splits['test'])} test)")
    else:
        sys.exit(f"{path}: differs from the rebuilt splits")


if __name__ == "__main__":
    main()
