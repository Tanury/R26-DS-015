"""
02_extract_slices.py
─────────────────────

Extracts the optimal transaxial slice from each reconstructed DaTscan
NIfTI volume and saves as a normalised 224×224 numpy array.

Volume shape: (91, 109, 91) at 2mm³ isotropic
  - Axis 0 = 91 slices  (sagittal)
  - Axis 1 = 109 slices (axial/transaxial) ← correct plane for striatum
  - Axis 2 = 91 slices  (coronal)

Standard approach (PPMI protocol + literature):
  - Slice along axis 1 (transaxial)
  - Peak striatal uptake around slice 41 (mid-striatum)
  - Average ±4 slices around peak for robustness

Intensity normalisation: SBR-style using occipital reference region

Input : data/nifti/dat_recon/<subject_id>.nii.gz
Output: data/preprocessed/dat_slices/<subject_id>_slice.npy  (float32, 224×224)

Usage:
    python preprocessing/dat/02_extract_slices.py
"""

import logging
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ = parkinsons/preprocessing/dat/02_extract_slices.py
# parents[0] = dat/
# parents[1] = preprocessing/
# parents[2] = parkinsons/  ← ROOT
ROOT       = Path(__file__).resolve().parents[2]
NIFTI_DIR  = ROOT / 'data' / 'nifti' / 'dat_recon'
SLICES_DIR = ROOT / 'data' / 'preprocessed' / 'dat_slices'
SLICES_DIR.mkdir(parents=True, exist_ok=True)

RESIZE_DIM     = 224
EXPECTED_SHAPE = (91, 109, 91)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def extract_slice(data: np.ndarray):
    """
    Extract best transaxial slice from a (91, 109, 91) DaTscan volume.
    Axis 1 = transaxial (109 slices) — top-down view showing bilateral striatum.
    """
    lo = int(data.shape[1] * 0.30)   # ~33
    hi = int(data.shape[1] * 0.70)   # ~76

    slice_means = data[:, lo:hi, :].mean(axis=(0, 2))
    best_local  = int(np.argmax(slice_means))
    best_ax1    = lo + best_local

    w_lo     = max(0, best_ax1 - 4)
    w_hi     = min(data.shape[1], best_ax1 + 5)
    slice_2d = data[:, w_lo:w_hi, :].mean(axis=1)   # (91, 91)

    return slice_2d, best_ax1


def normalise(slice_2d: np.ndarray) -> np.ndarray:
    """SBR-style normalisation using posterior reference region."""
    resized = cv2.resize(
        slice_2d, (RESIZE_DIM, RESIZE_DIM),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    h          = resized.shape[0]
    ref_region = resized[int(h * 0.75):, :]
    ref_vals   = ref_region[ref_region > 0]

    if len(ref_vals) > 10:
        ref_mean   = ref_vals.mean()
        normalised = resized / (ref_mean + 1e-8)
    else:
        nonzero = resized[resized > 0]
        if len(nonzero) > 0:
            p_low  = np.percentile(nonzero, 1)
            p_high = np.percentile(nonzero, 99)
            normalised = np.clip(resized, p_low, p_high)
            if p_high > p_low:
                normalised = (normalised - p_low) / (p_high - p_low)
        else:
            normalised = resized

    return normalised.astype(np.float32)


def main():
    log.info(f'ROOT      : {ROOT}')
    log.info(f'NIFTI_DIR : {NIFTI_DIR}')

    all_niftis = sorted(NIFTI_DIR.glob('*.nii.gz'))
    log.info(f'Found {len(all_niftis)} NIfTI files')

    ok       = 0
    skipped  = 0
    failed   = 0
    excluded = []

    for nii in all_niftis:
        subject_id = nii.stem.replace('.nii', '')
        out_path   = SLICES_DIR / f'{subject_id}_slice.npy'

        if out_path.exists():
            skipped += 1
            continue

        try:
            img  = nib.load(str(nii))
            data = img.get_fdata(dtype=np.float32)

            # Handle 4D — take first frame only
            if data.ndim == 4:
                data = data[..., 0]

            # Exclude wrong shapes
            if data.shape != EXPECTED_SHAPE:
                log.warning(f'Excluding {subject_id} — shape {data.shape}')
                excluded.append(subject_id)
                continue

            slice_2d, best_ax1 = extract_slice(data)
            normalised         = normalise(slice_2d)

            np.save(str(out_path), normalised)
            ok += 1

            if ok % 100 == 0:
                log.info(f'  Progress: {ok} done (last best_ax1={best_ax1})')

        except Exception as e:
            log.error(f'FAILED {subject_id}: {e}')
            failed += 1

    log.info('')
    log.info('── Slice extraction complete ─────────────────')
    log.info(f'  Extracted : {ok}')
    log.info(f'  Skipped   : {skipped}  (already existed)')
    log.info(f'  Excluded  : {len(excluded)}  (wrong shape: {excluded})')
    log.info(f'  Failed    : {failed}')
    log.info(f'  Output    : {SLICES_DIR}')


if __name__ == '__main__':
    main()
