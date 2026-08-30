"""
02_dat_dicom_to_nifti.py
─────────────────────────
Run LOCALLY on MacBook.

Converts reconstructed DaTscan SPECT DICOMs → NIfTI (.nii.gz)
using dcm2niix.

Input : data/raw/dat_recon_dicom/PPMI/<subject_id>/Reconstructed_DaTSCAN/<date>/<id>/*.dcm
Output: data/nifti/dat_recon/<subject_id>.nii.gz

Usage:
    python preprocessing/02_dat_dicom_to_nifti.py
"""

import subprocess
import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
PPMI_ROOT = ROOT / 'data' / 'raw' / 'dat_recon_dicom' / 'PPMI'
NIFTI_DIR = ROOT / 'data' / 'nifti' / 'dat_recon'
NIFTI_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def main():
    subject_dirs = sorted([d for d in PPMI_ROOT.iterdir() if d.is_dir()])
    log.info(f'Found {len(subject_dirs)} subjects in {PPMI_ROOT}')

    converted = 0
    skipped   = 0
    failed    = 0

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        out_path   = NIFTI_DIR / f'{subject_id}.nii.gz'

        if out_path.exists():
            skipped += 1
            continue

        dcm_files = list(subj_dir.rglob('*.dcm'))
        if not dcm_files:
            log.warning(f'No DICOM found: {subject_id}')
            failed += 1
            continue

        # Point dcm2niix at the folder containing the .dcm file
        dcm_folder = dcm_files[0].parent

        result = subprocess.run(
            ['dcm2niix', '-z', 'y', '-f', subject_id,
             '-o', str(NIFTI_DIR), str(dcm_folder)],
            capture_output=True, text=True
        )

        # dcm2niix may append series info to filename — find what it created
        candidates = [
            f for f in NIFTI_DIR.glob(f'{subject_id}*.nii.gz')
            if 'mask' not in f.name.lower()
        ]

        if out_path.exists():
            converted += 1
        elif candidates:
            best = max(candidates, key=lambda f: f.stat().st_size)
            best.rename(out_path)
            converted += 1
        else:
            log.error(f'FAILED {subject_id}: {result.stderr[:100]}')
            failed += 1

        total_done = converted + skipped
        if total_done % 100 == 0:
            log.info(f'  Progress: {total_done}/{len(subject_dirs)}')

    log.info('')
    log.info('── Conversion complete ──────────────────────────')
    log.info(f'  Converted : {converted}')
    log.info(f'  Skipped   : {skipped}  (already existed)')
    log.info(f'  Failed    : {failed}')
    log.info(f'  Output    : {NIFTI_DIR}')


if __name__ == '__main__':
    main()
