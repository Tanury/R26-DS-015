"""
backend/services/oct_visualization.py

Renders a "B-scan input vs delineated layers" comparison image for the
OCT MS dashboard result -- same idea as ms_visualization.py for MRI,
but for retinal OCT: shows the raw B-scan, then the same B-scan with
the 9 manual boundary curves overlaid and the 8 layers between them
shaded, making visible exactly what thickness.py measures.

Returns a base64-encoded PNG data URI, same interface as the MRI
visualization, so the frontend's existing slice_image display works
for both branches without any frontend changes.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# One color per layer (RNFL -> RPE, top to bottom), chosen for visual
# distinction against a grayscale B-scan background.
LAYER_COLORS = [
    (0.95, 0.55, 0.20, 0.45),   # RNFL
    (0.30, 0.70, 0.95, 0.45),   # GCIP
    (0.55, 0.85, 0.35, 0.45),   # INL
    (0.85, 0.35, 0.75, 0.45),   # OPL
    (0.95, 0.85, 0.25, 0.45),   # ONL
    (0.40, 0.40, 0.90, 0.45),   # IS
    (0.90, 0.30, 0.30, 0.45),   # OS
    (0.60, 0.40, 0.20, 0.45),   # RPE
]
BOUNDARY_COLOR = (1.0, 1.0, 1.0, 0.9)


def render_bscan_with_boundaries(
    bscan: np.ndarray, boundaries: np.ndarray, layer_names: list
) -> str:
    """
    Args:
        bscan: (size_y, size_x) raw B-scan pixel array (one central-window frame)
        boundaries: (n_boundaries, size_x) dense y-values for THIS SAME B-scan
                    (i.e. mat_loader's boundaries[frame_idx], not the whole volume)
        layer_names: the 8 layer names, top to bottom (thickness.LAYER_NAMES)

    Returns a base64 PNG data URI.
    """
    size_y, size_x = bscan.shape
    x_grid = np.arange(size_x)

    # Normalize B-scan intensity for display (percentile clip, same approach as MRI)
    finite = bscan[np.isfinite(bscan)]
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    gray = np.clip((bscan - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="#0f1117")
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, size_x)
        ax.set_ylim(size_y, 0)  # image-style y-axis (0 at top)

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Input B-scan", color="#cbd5e1", fontsize=11, pad=8)

    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=1, aspect="auto")
    n_layers = boundaries.shape[0] - 1
    for i in range(n_layers):
        top, bottom = boundaries[i], boundaries[i + 1]
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        axes[1].fill_between(x_grid, top, bottom, color=color, linewidth=0)
    for b in boundaries:
        axes[1].plot(x_grid, b, color=BOUNDARY_COLOR, linewidth=0.6)
    axes[1].set_title("Delineated Layers", color="#cbd5e1", fontsize=11, pad=8)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=LAYER_COLORS[i % len(LAYER_COLORS)])
               for i in range(len(layer_names))]
    fig.legend(handles, layer_names, loc="lower center", ncol=len(layer_names),
               fontsize=8, frameon=False, labelcolor="#cbd5e1", bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"