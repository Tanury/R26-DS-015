"""
04_filter_quality.py

Reads the quality report and removes flagged subjects from the
DaTscan slice directory, so they don't pollute training.

Usage:
    python preprocessing/dat/05_filter_quality.py

Inputs:
    logs/datscan_quality_report.csv
    data/preprocessed/dat_slices/

Outputs:
    data/preprocessed/dat_slices_clean/   ← only clean subjects
    logs/datscan_excluded_subjects.txt    ← list of removed subject IDs
"""

import csv
import shutil
import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SLICES_DIR  = ROOT / 'data' / 'preprocessed' / 'dat_slices'
CLEAN_DIR   = ROOT / 'data' / 'preprocessed' / 'dat_slices_clean'
REPORT_CSV  = ROOT / 'logs' / 'datscan_quality_report.csv'
EXCLUDE_TXT = ROOT / 'logs' / 'datscan_excluded_subjects.txt'
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def main():
    log.info(f'ROOT      : {ROOT}')
    log.info(f'Report    : {REPORT_CSV}')

    if not REPORT_CSV.exists():
        log.error('Quality report not found — run 03_visual_check.py first')
        return

    # Load quality report
    good_subjects    = []
    exclude_subjects = []

    with open(REPORT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['flag'] == 'ok':
                good_subjects.append(row['subject_id'])
            else:
                exclude_subjects.append((row['subject_id'], row['flag']))

    log.info(f'Good subjects    : {len(good_subjects)}')
    log.info(f'Excluded subjects: {len(exclude_subjects)}')

    # Copy only clean slices to clean dir
    copied  = 0
    missing = 0

    for subject_id in good_subjects:
        src = SLICES_DIR / f'{subject_id}_slice.npy'
        dst = CLEAN_DIR  / f'{subject_id}_slice.npy'

        if dst.exists():
            copied += 1
            continue

        if src.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1
        else:
            log.warning(f'Slice not found: {subject_id}')
            missing += 1

    # Write exclusion list
    with open(EXCLUDE_TXT, 'w') as f:
        f.write('# Subjects excluded from DaTscan training due to quality issues\n')
        f.write('# Format: subject_id, reason\n\n')
        for subject_id, reason in exclude_subjects:
            f.write(f'{subject_id},{reason}\n')

    log.info('')
    log.info('── Filter complete ───────────────────────────')
    log.info(f'  Copied to clean dir : {copied}')
    log.info(f'  Missing slices      : {missing}')
    log.info(f'  Excluded            : {len(exclude_subjects)}')
    log.info(f'  Clean dir           : {CLEAN_DIR}')
    log.info(f'  Exclusion list      : {EXCLUDE_TXT}')
    log.info('')
    log.info('  Update dat_encoder dataset.py to use dat_slices_clean/')


if __name__ == '__main__':
    main()
