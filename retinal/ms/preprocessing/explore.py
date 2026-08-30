"""
step 01 of MS Retinal OCT Preprocessing 
explore.py
─────────────
Explores the HC-MS OCT dataset structure and produces
a visual sanity check of one subject's B-scans and layer boundaries.

HC-MS / Johns Hopkins IACL
  - 14 HC + 21 MS subjects
  - 49 B-scans per subject (496 depth × 1024 width)
  - 11 layer boundary curves per B-scan (21 control points each)

"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from OCTVol import OCTVol

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / 'retinal' / 'ms' / 'data' / 'raw' / 'OCT_Manual_Delineations'
VOL_DIR = RAW_DIR / 'vol'
MAT_DIR = RAW_DIR / 'delineation'
LOG_DIR = ROOT / 'retinal' / 'ms' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_vol(path: Path) -> np.ndarray:
    """Load .vol file → (49, 496, 1024) float32 array."""
    vol    = OCTVol(str(path))
    bscans = vol.b_scans                  # (496, 1024, 49)
    return bscans.transpose(2, 0, 1)     # → (49, 496, 1024)


def load_boundaries(path: Path) -> np.ndarray:
    """
    Load .mat file → (49, 11, 1024) float32 boundary array.
    Each of the 11 boundaries is interpolated to full 1024-pixel width.
    """
    m          = sio.loadmat(str(path))
    ctrl_pts   = m['control_pts']        # (49, 11) object array
    num_bscans = ctrl_pts.shape[0]
    num_layers = ctrl_pts.shape[1]
    width      = 1024

    boundaries = np.full((num_bscans, num_layers, width), np.nan, dtype=np.float32)

    for b in range(num_bscans):
        for l in range(num_layers):
            pts = ctrl_pts[b, l]         # (21, 2) — (x, y) control points
            if pts is None or not hasattr(pts, 'shape') or pts.size == 0:
                continue
            xs = pts[:, 0].astype(float)
            ys = pts[:, 1].astype(float)
            # Interpolate to full width
            xi = np.arange(1, width + 1, dtype=float)
            boundaries[b, l] = np.interp(xi, xs, ys)

    return boundaries


def normalise_bscan(bscan: np.ndarray) -> np.ndarray:
    """Normalise a single B-scan to [0, 1] using percentile clipping."""
    p_low  = np.percentile(bscan, 1)
    p_high = np.percentile(bscan, 99)
    clipped = np.clip(bscan, p_low, p_high)
    if p_high > p_low:
        return (clipped - p_low) / (p_high - p_low)
    return clipped


def make_3channel(bscan: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Build 3-channel composite image from one B-scan + its layer boundaries.

    Channel 0: raw grayscale B-scan (normalised)
    Channel 1: layer-masked image (boundaries overlaid as coloured regions)
    Channel 2: binary contour map (boundaries as binary lines)

    Returns: (3, H, W) float32 array in [0, 1]
    """
    H, W   = bscan.shape
    raw    = normalise_bscan(bscan)

    # Channel 1 — layer mask: colour each inter-boundary region differently
    layer_mask = raw.copy()
    num_layers = boundaries.shape[0]
    for l in range(num_layers - 1):
        top_boundary = boundaries[l].astype(int).clip(0, H - 1)
        bot_boundary = boundaries[l + 1].astype(int).clip(0, H - 1)
        weight = 0.15 + 0.05 * l   # slight intensity modulation per layer
        for x in range(W):
            y_top = top_boundary[x]
            y_bot = bot_boundary[x]
            if y_bot > y_top:
                layer_mask[y_top:y_bot, x] = np.clip(
                    layer_mask[y_top:y_bot, x] + weight, 0, 1
                )

    # Channel 2 — binary contour: 1 pixel wide lines at boundary positions
    contour = np.zeros((H, W), dtype=np.float32)
    for l in range(num_layers):
        ys = boundaries[l].astype(int).clip(0, H - 1)
        for x in range(W):
            contour[ys[x], x] = 1.0

    return np.stack([raw, layer_mask, contour], axis=0)   # (3, H, W)


def main():
    vols = sorted(VOL_DIR.glob('*.vol'))
    mats = sorted(MAT_DIR.glob('*.mat'))
    print(f'VOL files: {len(vols)}  MAT files: {len(mats)}')

    # Load first subject
    sample_id  = vols[0].stem
    print(f'\nExploring: {sample_id}')
    volume     = load_vol(vols[0])
    boundaries = load_boundaries(mats[0])   # (49, 11, 1024)

    print(f'Volume shape    : {volume.shape}   (B-scans × depth × width)')
    print(f'Boundaries shape: {boundaries.shape}   (B-scans × layers × width)')
    print(f'Volume dtype    : {volume.dtype}')
    print(f'Volume min/max  : {volume.min():.4f} / {volume.max():.4f}')
    print(f'Boundary y range: {np.nanmin(boundaries):.1f} – {np.nanmax(boundaries):.1f} pixels')

    # Build 3-channel composite for the middle B-scan
    mid_b  = 24
    bscan  = volume[mid_b]               # (496, 1024)
    bounds = boundaries[mid_b]           # (11, 1024)
    comp   = make_3channel(bscan, bounds) # (3, 496, 1024)

    print(f'\n3-channel composite shape: {comp.shape}')

    # ── Visualise ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1 — three channels
    titles = ['Ch0: Raw grayscale', 'Ch1: Layer masked', 'Ch2: Binary contour']
    cmaps  = ['gray', 'gray', 'gray']
    for col in range(3):
        axes[0, col].imshow(comp[col], cmap=cmaps[col], aspect='auto', origin='upper')
        axes[0, col].set_title(f'{titles[col]}\n{sample_id} B-scan {mid_b}', fontsize=10)
        axes[0, col].axis('off')

    # Row 2 — raw B-scan with all 11 boundaries overlaid
    axes[1, 0].imshow(normalise_bscan(bscan), cmap='gray', aspect='auto', origin='upper')
    colors = cm.rainbow(np.linspace(0, 1, 11))
    for l in range(11):
        axes[1, 0].plot(bounds[l], color=colors[l], linewidth=0.8, label=f'L{l}')
    axes[1, 0].set_title(f'All 11 layer boundaries — B-scan {mid_b}', fontsize=10)
    axes[1, 0].legend(fontsize=6, loc='upper right')
    axes[1, 0].axis('off')

    # Row 2 — 5 consecutive B-scans (raw)
    for col, b_idx in enumerate([20, 22, 24, 26, 28]):
        if col >= 2: break  # only 2 slots left
        ax = axes[1, col + 1]
        ax.imshow(normalise_bscan(volume[b_idx]), cmap='gray', aspect='auto', origin='upper')
        ax.set_title(f'B-scan {b_idx}', fontsize=9)
        ax.axis('off')

    # Fill remaining subplots
    axes[1, 2].imshow(normalise_bscan(volume[28]), cmap='gray', aspect='auto', origin='upper')
    axes[1, 2].set_title('B-scan 28', fontsize=9)
    axes[1, 2].axis('off')

    plt.suptitle(f'HC-MS OCT Dataset — {sample_id} exploration', y=1.01, fontsize=13)
    plt.tight_layout()

    out_path = LOG_DIR / 'ms_oct_explore.png'
    plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
    plt.show()
    print(f'\n✓ Saved to {out_path}')


if __name__ == '__main__':
    main()
