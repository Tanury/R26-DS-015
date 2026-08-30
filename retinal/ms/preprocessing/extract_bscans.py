
"""
01_extract_bscans.py

Extracts all 49 B-scan frames from each .vol file and saves as
normalised PNG images. Also reads .mat boundary files and saves
boundary arrays as .npy files for the 3-channel composite step.

Dataset: HC-MS / Johns Hopkins IACL (35 subjects: 14 HC + 21 MS)
Each subject: 49 B-scans × 496 × 1024 pixels

Output structure:
  data/preprocessed/bscans/
    hc01/
      frame_000.png  ...  frame_048.png
      boundaries.npy     ← (49, 11, 1024) interpolated boundaries
    hc02/ ...
    ms01/ ...

Usage:
    python retinal/ms/preprocessing/01_extract_bscans.py
"""

import numpy as np
import scipy.io as sio
import cv2
import logging
from pathlib import Path
from OCTVol import OCTVol

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / 'retinal' / 'ms' / 'data' / 'raw' / 'OCT_Manual_Delineations-2018_June_29'
VOL_DIR = RAW_DIR / 'vol'
MAT_DIR = RAW_DIR / 'delineation'
OUT_DIR = ROOT / 'retinal' / 'ms' / 'data' / 'preprocessed' / 'bscans'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224   # resize B-scans to 224×224 for CNN input

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def load_bscans(vol_path: Path) -> np.ndarray:
    """
    Load .vol file → (49, 496, 1024) float32 array.
    OCTVol returns (496, 1024, 49) — we transpose to (49, 496, 1024).
    """
    vol    = OCTVol(str(vol_path))
    bscans = vol.b_scans           # (496, 1024, 49)
    return bscans.transpose(2, 0, 1).astype(np.float32)  # (49, 496, 1024)


def normalise_bscan(bscan: np.ndarray) -> np.ndarray:
    """
    Normalise Spectralis OCT float32 B-scan to uint8 [0, 255].
    Step 1: Remove sentinel values (float32 max ~3.4e38 used as NaN).
    Step 2: Log compression — standard for OCT display.
    Step 3: Percentile clip and scale to [0, 255].
    """
    SENTINEL = 3.0e38
    clean = bscan.copy()
    clean[clean >= SENTINEL] = 0.0
    log_img = np.log1p(clean * 1000)
    p_low  = np.percentile(log_img, 1)
    p_high = np.percentile(log_img, 99)
    clipped = np.clip(log_img, p_low, p_high)
    if p_high > p_low:
        norm = (clipped - p_low) / (p_high - p_low)
    else:
        norm = np.zeros_like(clipped)
    return (norm * 255).astype(np.uint8)


def load_boundaries(mat_path: Path, width: int = 1024) -> np.ndarray:
    """
    Load .mat boundary file → (49, N, 1024) float32 array.
    Handles two .mat formats found in the HC-MS dataset:

    Format A — control_pts: (49, 11) object array
      Each element: (21, 2) float64 control points, or (0,0) if missing
    Format B — bd_pts: (1024, 49, 9) float64
      Already fully interpolated — just transpose
    """
    m  = sio.loadmat(str(mat_path))
    xi = np.arange(1, width + 1, dtype=float)

    # Format B: bd_pts already interpolated
    if 'bd_pts' in m:
        bd = m['bd_pts'].astype(np.float32)   # (1024, 49, 9)
        return bd.transpose(1, 2, 0)           # → (49, 9, 1024)

    # Format A: control_pts with (x,y) control points
    ctrl_pts   = m['control_pts']
    num_bscans = ctrl_pts.shape[0]
    num_layers = ctrl_pts.shape[1]
    boundaries = np.full((num_bscans, num_layers, width), np.nan, dtype=np.float32)

    for b in range(num_bscans):
        for l in range(num_layers):
            pts = ctrl_pts[b, l]
            if pts is None or not hasattr(pts, 'shape') or pts.size == 0:
                continue
            if pts.ndim < 2 or pts.shape[1] < 2:
                continue
            xs = pts[:, 0].astype(float)
            ys = pts[:, 1].astype(float)
            if len(xs) < 2:
                continue
            order = np.argsort(xs)
            boundaries[b, l] = np.interp(xi, xs[order], ys[order])

    return boundaries


def main():
    vols = sorted(VOL_DIR.glob('*.vol'))
    mats = sorted(MAT_DIR.glob('*.mat'))

    log.info(f'ROOT    : {ROOT}')
    log.info(f'VOL files: {len(vols)}  MAT files: {len(mats)}')

    ok   = 0
    skip = 0
    fail = 0

    for vol_path in vols:
        subject_id = vol_path.stem   # e.g. hc01_spectralis_macula_v1_s1_R
        # Shorten to hc01 / ms01 etc.
        short_id   = subject_id.split('_')[0]
        subj_dir   = OUT_DIR / short_id
        subj_dir.mkdir(exist_ok=True)

        # Check if already done
        existing_pngs = list(subj_dir.glob('frame_*.png'))
        if len(existing_pngs) == 49 and (subj_dir / 'boundaries.npy').exists():
            log.info(f'  ✓ {short_id} — already extracted, skipping')
            skip += 1
            continue

        # Find matching .mat file
        mat_path = MAT_DIR / (subject_id + '.mat')
        if not mat_path.exists():
            log.warning(f'  ⚠ No .mat for {subject_id}')

        try:
            log.info(f'  Extracting {short_id}...')

            # Load B-scans
            volume = load_bscans(vol_path)   # (49, 496, 1024)

            # Save each B-scan as PNG
            for i, bscan in enumerate(volume):
                norm    = normalise_bscan(bscan)
                resized = cv2.resize(norm, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
                out_png = subj_dir / f'frame_{i:03d}.png'
                cv2.imwrite(str(out_png), resized)

            # Load and save boundaries
            if mat_path.exists():
                boundaries = load_boundaries(mat_path)   # (49, 11, 1024)
                np.save(str(subj_dir / 'boundaries.npy'), boundaries)
                log.info(f'    ✓ {short_id}: 49 frames + boundaries saved')
            else:
                log.warning(f'    ⚠ {short_id}: 49 frames saved, no boundaries')

            ok += 1

        except Exception as e:
            log.error(f'  ✗ FAILED {short_id}: {e}')
            fail += 1

    log.info('')
    log.info('── Extraction complete ──────────────────────────')
    log.info(f'  Extracted : {ok} subjects')
    log.info(f'  Skipped   : {skip} (already done)')
    log.info(f'  Failed    : {fail}')
    log.info(f'  Output    : {OUT_DIR}')
    log.info(f'  Total PNGs: {ok * 49} B-scan images')


if __name__ == '__main__':
    main()