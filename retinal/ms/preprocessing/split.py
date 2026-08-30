"""
02
split.py
────────────
Creates subject-level stratified train/val/test splits.

CRITICAL: Split at SUBJECT level, not B-scan level.
          All 49 B-scans from one subject stay in the same split.
          Prevents data leakage between train and test.

Dataset: 14 HC + 21 MS = 35 subjects
Split:   70% train / 15% val / 15% test (subject level)
         HC:  9 train / 2 val / 3 test
         MS: 15 train / 3 val / 3 test

Usage:
    python retinal/ms/preprocessing/02_split.py
"""

import json
import random
import csv
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).resolve().parents[3]
BSCAN_DIR = ROOT / 'retinal' / 'ms' / 'data' / 'preprocessed' / 'bscans'
SPLITS_DIR= ROOT / 'retinal' / 'ms' / 'data' / 'splits'
DEMO_CSV  = ROOT / 'retinal' / 'ms' / 'data' / 'raw' / \
            'OCT_Manual_Delineations-2018_June_29' / 'demographics-2018_June_29.csv'
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)


def get_label(subject_id: str) -> str:
    return 'HC' if subject_id.startswith('hc') else 'MS'


def main():
    # Get all extracted subjects
    all_subjects = sorted([
        d.name for d in BSCAN_DIR.iterdir()
        if d.is_dir() and len(list(d.glob('frame_*.png'))) == 49
    ])
    print(f'Total subjects with 49 frames: {len(all_subjects)}')

    # Group by label
    by_label = defaultdict(list)
    for s in all_subjects:
        by_label[get_label(s)].append(s)

    for label, subjects in by_label.items():
        print(f'  {label}: {len(subjects)} subjects — {subjects}')

    # Stratified split
    train_subjects, val_subjects, test_subjects = [], [], []

    for label, subjects in by_label.items():
        random.shuffle(subjects)
        n       = len(subjects)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        train_subjects += subjects[:n_train]
        val_subjects   += subjects[n_train:n_train + n_val]
        test_subjects  += subjects[n_train + n_val:]
        print(f'  {label}: {n_train} train / {n_val} val / {n-n_train-n_val} test')

    # Build B-scan level records (one entry per B-scan, but split at subject level)
    def build_records(subject_list):
        records = []
        for subj in sorted(subject_list):
            label     = get_label(subj)
            subj_dir  = BSCAN_DIR / subj
            for png in sorted(subj_dir.glob('frame_*.png')):
                records.append({
                    'subject_id':   subj,
                    'label':        label,
                    'frame':        png.stem,
                    'file_path':    str(png),
                    'boundary_path':str(subj_dir / 'boundaries.npy'),
                    'modality':     'OCT',
                    'source':       'HC-MS IACL',
                })
        return records

    train_records = build_records(train_subjects)
    val_records   = build_records(val_subjects)
    test_records  = build_records(test_subjects)

    # Save splits
    for name, records, subjects in [
        ('train', train_records, train_subjects),
        ('val',   val_records,   val_subjects),
        ('test',  test_records,  test_subjects),
    ]:
        out = SPLITS_DIR / f'ms_{name}.json'
        with open(out, 'w') as f:
            json.dump(records, f, indent=2)
        hc = sum(1 for r in records if r['label'] == 'HC')
        ms = sum(1 for r in records if r['label'] == 'MS')
        print(f'✓ {name}: {len(subjects)} subjects / {len(records)} B-scans '
              f'(HC={hc} MS={ms}) → {out.name}')

    print(f'\nTotal B-scans: {len(train_records)+len(val_records)+len(test_records)}')
    print(f'Train subjects: {sorted(train_subjects)}')
    print(f'Val subjects  : {sorted(val_subjects)}')
    print(f'Test subjects : {sorted(test_subjects)}')


if __name__ == '__main__':
    main()

