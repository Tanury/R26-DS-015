"""
backend/services/ms_visualization.py

Renders a side-by-side "input vs MindGlide segmentation" slice image for
the MS MRI dashboard result, similar in spirit to the comparison images
in MindGlide's own README -- but chosen dynamically per upload rather
than a fixed slice: picks whichever axial slice contains the most
Lesion tissue (if any), falling back to the slice with the largest
brain cross-section for a healthy/lesion-free scan.

Returns a base64-encoded PNG data URI, ready to drop directly into an
<img src="..."> tag on the frontend.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless -- no display backend needed on a server
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np

LESION_LABEL = 18

# One distinct, low-saturation color per MindGlide region (0=background is
# left transparent), with Lesion given a stark, high-contrast color since
# it's the most clinically important region to make immediately visible.
LABEL_COLORS = {
    0: (0, 0, 0, 0),                    # Background -- fully transparent
    1: (0.55, 0.75, 0.90, 0.55),        # CSF
    2: (0.30, 0.55, 0.85, 0.55),        # Ventricles_3_4_5
    3: (0.85, 0.65, 0.35, 0.55),        # DGM
    4: (0.75, 0.45, 0.75, 0.55),        # Pons
    5: (0.65, 0.35, 0.65, 0.55),        # Brainstem
    6: (0.40, 0.75, 0.55, 0.55),        # Cerebellum
    7: (0.90, 0.70, 0.50, 0.55),        # Temporal_lobe
    8: (0.35, 0.60, 0.90, 0.55),        # Temporal_horn_lateral_ventricle
    9: (0.25, 0.45, 0.80, 0.55),        # Lateral_ventricle
    10: (0.80, 0.80, 0.40, 0.55),       # Optic_chiasm
    11: (0.50, 0.80, 0.65, 0.55),       # Cerebellar_vermis
    12: (0.95, 0.85, 0.55, 0.55),       # Corpus_callosum
    13: (0.70, 0.70, 0.75, 0.45),       # White_matter
    14: (0.95, 0.60, 0.60, 0.55),       # Frontal_lobe_GM
    15: (0.80, 0.55, 0.85, 0.55),       # Limbic_cortex_GM
    16: (0.60, 0.85, 0.90, 0.55),       # Parietal_lobe_GM
    17: (0.85, 0.75, 0.95, 0.55),       # Occipital_lobe_GM
    18: (1.00, 0.10, 0.10, 0.85),       # Lesion -- stark red, high opacity, stands out deliberately
    19: (0.55, 0.45, 0.35, 0.55),       # Ventral_diencephalon
}


def _find_best_slice(seg_data: np.ndarray) -> int:
    """Pick the axial slice with the most Lesion tissue, or if there's no
    lesion at all (a healthy scan), the slice with the largest brain
    cross-section -- more informative than an arbitrary fixed slice."""
    lesion_counts = (seg_data == LESION_LABEL).sum(axis=(0, 1))
    if lesion_counts.max() > 0:
        return int(np.argmax(lesion_counts))
    brain_counts = (seg_data > 0).sum(axis=(0, 1))
    return int(np.argmax(brain_counts))


def _normalize_grayscale(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        return np.zeros_like(slice_2d)
    clipped = np.clip(slice_2d, lo, hi)
    return (clipped - lo) / (hi - lo)


def render_input_vs_segmentation(input_path: str, seg_path: str) -> str:
    """Returns a base64 PNG data URI (str, ready for <img src=...>) showing
    the raw input slice next to the same slice with the MindGlide
    segmentation overlaid."""
    input_img = nib.load(input_path)
    seg_img = nib.load(seg_path)
    input_data = input_img.get_fdata()
    seg_data = seg_img.get_fdata()

    slice_idx = _find_best_slice(seg_data)
    input_slice = np.rot90(input_data[:, :, slice_idx])
    seg_slice = np.rot90(seg_data[:, :, slice_idx])

    gray = _normalize_grayscale(input_slice)

    overlay = np.zeros((*seg_slice.shape, 4))
    for label, color in LABEL_COLORS.items():
        overlay[seg_slice == label] = color

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor="#0f1117")
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input MRI", color="#cbd5e1", fontsize=11, pad=8)

    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(overlay)
    axes[1].set_title("MindGlide Segmentation", color="#cbd5e1", fontsize=11, pad=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"