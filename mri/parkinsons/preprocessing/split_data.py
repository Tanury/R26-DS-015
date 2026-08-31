"""

Reads:  data/compiled/labels.csv
Writes: data/splits/train.json
        data/splits/val.json
        data/splits/test.json

Split ratios: 70% train / 15% val / 15% test
Stratified by: label (PD/HC) within each modality (MRI / DaTscan)

Usage:
  python preprocessing/06_split_data.py
"""

import json
import logging
import random
import csv
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).resolve().parents[1]
LABELS    = ROOT / "data" / "compiled" / "labels.csv"
SPLITS    = ROOT / "data" / "splits"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    random.seed(SEED)
    SPLITS.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(LABELS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by modality + label for stratified split
    groups = defaultdict(list)
    for row in rows:
        key = (row["modality"], row["label"])
        groups[key].append(row)

    train, val, test = [], [], []

    for (modality, label), group_rows in groups.items():
        random.shuffle(group_rows)
        n       = len(group_rows)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        train += group_rows[:n_train]
        val   += group_rows[n_train:n_train + n_val]
        test  += group_rows[n_train + n_val:]

        log.info(f"  {modality} {label}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
        out_path = SPLITS / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_rows, f, indent=2)
        log.info(f"  Wrote {len(split_rows)} rows → {out_path}")

    log.info(f"\n  Total: {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()
