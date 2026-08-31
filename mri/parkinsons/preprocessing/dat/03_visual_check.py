"""
04_dat_visual_check.py
───────────────────────
Generates a visual sample of extracted DaTscan slices and runs
a quality check across all subjects, flagging low-signal and
asymmetric cases for exclusion.

Usage:
    python preprocessing/dat/03_visual_check.py

Outputs:
    logs/datscan_recon_final.png      — sample grid (8 subjects)
    logs/datscan_quality_report.csv   — per-subject quality flags
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import logging
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ = parkinsons/preprocessing/dat/04_dat_visual_check.py
# parents[0] = dat/
# parents[1] = preprocessing/
# parents[2] = parkinsons/  ← ROOT
ROOT       = Path(__file__).resolve().parents[2]
SLICES_DIR = ROOT / 'data' / 'preprocessed' / 'dat_slices'
LOG_DIR    = ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ── Visual sample ──────────────────────────────────────────────────────────────

def visual_check(samples: list, out_path: Path):
    indices = [0, 50, 100, 200, 300, 400, 500, 570]
    picks   = [samples[i] for i in indices if i < len(samples)]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, sample in zip(axes, picks):
        data = np.load(str(sample))
        ax.imshow(data, cmap='hot', origin='lower')
        ax.set_title(sample.stem[:12], fontsize=9)
        ax.axis('off')

    plt.suptitle(
        'DaTscan SPECT — Reconstructed | Axis 1 transaxial | SBR normalised (224×224)',
        y=1.01
    )
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
    plt.show()
    log.info(f'Visual sample saved → {out_path}')


# ── Quality check ──────────────────────────────────────────────────────────────

def quality_check(samples: list, report_path: Path) -> list:
    """
    Flags subjects with:
      - low_signal   : max/mean ratio < 2.0  (near-empty or noise-only)
      - asymmetric   : left/right intensity ratio < 0.5 (brain off-centre)
      - not_centred  : edge mean > centre mean

    Returns list of subject IDs to exclude.
    """
    results  = []
    issues   = []
    good     = 0

    for s in samples:
        subject_id = s.stem.replace('_slice', '')
        data       = np.load(str(s))
        h, w       = data.shape

        centre       = data[h//4:3*h//4, w//4:3*w//4].mean()
        edge         = np.concatenate([
            data[:h//4, :].flatten(),
            data[3*h//4:, :].flatten(),
            data[:, :w//4].flatten(),
            data[:, 3*w//4:].flatten()
        ]).mean()

        signal_ratio = data.max() / (data.mean() + 1e-8)
        left_mean    = data[:, :w//2].mean()
        right_mean   = data[:, w//2:].mean()
        symmetry     = min(left_mean, right_mean) / (max(left_mean, right_mean) + 1e-8)

        if signal_ratio < 2.0:
            flag = 'low_signal'
        elif symmetry < 0.5:
            flag = 'asymmetric'
        elif centre < edge:
            flag = 'not_centred'
        else:
            flag = 'ok'
            good += 1

        results.append({
            'subject_id':   subject_id,
            'flag':         flag,
            'signal_ratio': round(float(signal_ratio), 3),
            'symmetry':     round(float(symmetry), 3),
            'centre_mean':  round(float(centre), 4),
            'edge_mean':    round(float(edge), 4),
        })

        if flag != 'ok':
            issues.append(subject_id)

    # Write CSV report
    with open(report_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    issue_types = Counter(r['flag'] for r in results if r['flag'] != 'ok')

    log.info('')
    log.info('── Quality check results ─────────────────────')
    log.info(f'  Total subjects   : {len(samples)}')
    log.info(f'  Good             : {good}  ({good / len(samples) * 100:.1f}%)')
    log.info(f'  Issues           : {len(issues)}  ({len(issues) / len(samples) * 100:.1f}%)')
    log.info(f'  Issue breakdown  : {dict(issue_types)}')
    log.info(f'  Report saved     → {report_path}')

    return issues


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_slices = sorted(SLICES_DIR.glob('*.npy'))
    log.info(f'ROOT       : {ROOT}')
    log.info(f'SLICES_DIR : {SLICES_DIR}')
    log.info(f'Total slices: {len(all_slices)}')

    if not all_slices:
        log.error('No slices found — run 02_extract_slices.py first')
        return

    # Visual sample
    visual_check(
        all_slices,
        out_path=LOG_DIR / 'datscan_recon_final.png'
    )

    # Quality report
    exclude_list = quality_check(
        all_slices,
        report_path=LOG_DIR / 'datscan_quality_report.csv'
    )

    log.info('')
    log.info(f'  Subjects to exclude from training: {len(exclude_list)}')
    if exclude_list[:10]:
        log.info(f'  First 10: {exclude_list[:10]}')
    log.info('')
    log.info('  Next step: run preprocessing/01_compile_dataset.py')
    log.info('  The quality report will be used to filter labels.csv')


if __name__ == '__main__':
    main()
