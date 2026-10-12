"""Load a dataset: its messages, its taxonomy and its fixed splits.

    from data import load
    ds = load("banking77")
    ds.rows["b77-test-0"]   # ("How do I locate my card?", "card_arrival")
    ds.split("calib")       # [(id, text, intent), ...]

The messages are downloaded once into data/ at the repo root, from pinned commits, and checked
against their SHA-256. The taxonomy and splits come from datasets/ (see datasets/README.md).
Standard library only.
"""

import csv
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/{}"
BANKING77_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
CLINC150 = "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_oos_plus.json"
CLINC150_SHA256 = "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f"


@dataclass
class Dataset:
    rows: dict[str, tuple[str, str]]  # id -> (text, intent)
    taxonomy: dict                    # datasets/<name>/taxonomy.json
    splits: dict                      # datasets/<name>/splits.json: calib, test, shots (row IDs)

    def split(self, name: str) -> list[tuple[str, str, str]]:
        return [(i, *self.rows[i]) for i in self.splits[name]]

    def shots(self) -> dict[str, list[str]]:
        """The example messages the prompt shows under each intent."""
        return {intent: [self.rows[i][0] for i in ids] for intent, ids in self.splits["shots"].items()}


def load(name: str) -> Dataset:
    rows = {"banking77": _banking77, "clinc150": _clinc150}[name]()
    folder = ROOT / "datasets" / name
    return Dataset(rows, json.loads((folder / "taxonomy.json").read_text()),
                   json.loads((folder / "splits.json").read_text()))


def _fetch(url: str, sha256: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        sys.exit(f"{dest} isn't the file the series used. Delete it and run again.")
    return dest


def _banking77() -> dict[str, tuple[str, str]]:
    """ID = b77-<split>-<row>, counting rows from 0 after the header."""
    rows = {}
    for split in ("train", "test"):
        path = _fetch(BANKING77.format(f"{split}.csv"), BANKING77_SHA256[f"{split}.csv"],
                      ROOT / "data" / "banking77" / f"{split}.csv")
        with path.open(newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                rows[f"b77-{split}-{i}"] = (row["text"], row["category"])
    return rows


def _clinc150() -> dict[str, tuple[str, str]]:
    """ID = clinc-<split>-<row>; each split lists its in-scope rows first, then the out-of-scope ones."""
    path = _fetch(CLINC150, CLINC150_SHA256, ROOT / "data" / "clinc150" / "data_oos_plus.json")
    raw = json.loads(path.read_text())
    rows = {}
    for split, parts in (("train", ("train", "oos_train")), ("validation", ("val", "oos_val")), ("test", ("test", "oos_test"))):
        for i, (text, intent) in enumerate(r for p in parts for r in raw[p]):
            rows[f"clinc-{split}-{i}"] = (text, intent)
    return rows
