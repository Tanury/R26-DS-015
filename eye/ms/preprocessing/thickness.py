"""
thickness.py
============
R26-DS-015 — eye/ms preprocessing

Computes per-retinal-layer thickness from the dense boundary curves
produced by mat_loader.py, converted to physical units (microns) using
the paired .vol file's axial pixel scale (scale_y).

This is the reimplemented "useful idea" from the OCT/MS literature
(retinal layer thickness as a biomarker) — written from scratch against
this dataset's actual boundary format, not copied from any repository.

9 boundaries -> 8 layers, confirmed against He et al. (Data in Brief,
2019) and cross-checked against this dataset's actual .mat structure:
RNFL, GCIP (GCL+IPL), INL, OPL, ONL, IS, OS, RPE (top to bottom).

Usage:
    from thickness import compute_thickness_features
    feats = compute_thickness_features(boundaries, scale_y_mm=0.00387)
    feats.per_scan_um       # (n_bscans, 8) mean thickness per layer per B-scan, microns
    feats.subject_vector    # (8,) mean thickness per layer across all B-scans -- feeds ThicknessEncoder
    feats.layer_names       # ["RNFL", "GCIP", "INL", "OPL", "ONL", "IS", "OS", "RPE"]
"""

from dataclasses import dataclass, field

import numpy as np

LAYER_NAMES = ["RNFL", "GCIP", "INL", "OPL", "ONL", "IS", "OS", "RPE"]


@dataclass
class ThicknessFeatures:
    per_scan_um: np.ndarray     # (n_bscans, n_layers) microns
    subject_vector: np.ndarray  # (n_layers,) microns, mean over all B-scans/columns
    layer_names: list = field(default_factory=lambda: list(LAYER_NAMES))


def compute_thickness_features(
    boundaries: np.ndarray, scale_y_mm: float, bscan_indices=None
) -> ThicknessFeatures:
    """
    Args:
        boundaries: (n_bscans, n_boundaries, size_x) dense y-values in pixels,
                    as returned by mat_loader.load_boundaries(). Boundaries
                    must be ordered top-to-bottom (increasing y = increasing depth).
        scale_y_mm: mm per pixel along the axial (depth) direction, from the
                    paired .vol file's meta["scale_y"].
        bscan_indices: optional iterable of B-scan indices to restrict the
                    subject_vector average to (e.g. a central/foveal window).
                    per_scan_um always covers ALL B-scans regardless -- only
                    the subject-level average is restricted. None = use all
                    B-scans (default, backward compatible).

    Returns:
        ThicknessFeatures with per-B-scan and subject-level thickness vectors,
        in microns.
    """
    n_bscans, n_boundaries, size_x = boundaries.shape
    n_layers = n_boundaries - 1

    # thickness[i, l, x] = boundary[l+1] - boundary[l]  (pixels)
    thickness_px = np.diff(boundaries, axis=1)  # (n_bscans, n_layers, size_x)

    # negative values indicate boundary crossing/ordering error in the source
    # annotation for that column -- mask rather than silently keep
    thickness_px = np.where(thickness_px < 0, np.nan, thickness_px)

    scale_y_um = scale_y_mm * 1000.0
    thickness_um = thickness_px * scale_y_um  # (n_bscans, n_layers, size_x)

    per_scan_um = np.nanmean(thickness_um, axis=2)      # (n_bscans, n_layers) -- always all scans

    if bscan_indices is None:
        subject_vector = np.nanmean(per_scan_um, axis=0)
    else:
        subject_vector = np.nanmean(per_scan_um[list(bscan_indices)], axis=0)

    return ThicknessFeatures(
        per_scan_um=per_scan_um, subject_vector=subject_vector, layer_names=list(LAYER_NAMES)
    )


def central_bscan_window(n_bscans: int, window: int = 10) -> list:
    """Indices of the central `window` B-scans, centered on the middle
    scan (the foveal-region convention used in macular OCT thickness
    analysis -- e.g. the ETDRS grid focuses on central subfields)."""
    center = n_bscans // 2
    half = window // 2
    start = max(0, center - half)
    end = min(n_bscans, start + window)
    start = max(0, end - window)  # re-clamp if we hit the upper edge
    return list(range(start, end))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from mat_loader import load_boundaries
    from vol_loader import load_vol

    vol = load_vol(sys.argv[1])
    bd = load_boundaries(sys.argv[2], size_x=vol.meta["size_x"])

    feats = compute_thickness_features(bd.boundaries, scale_y_mm=vol.meta["scale_y"])
    print(f"per_scan_um shape: {feats.per_scan_um.shape}")
    for name, val in zip(feats.layer_names, feats.subject_vector):
        print(f"  {name:6s}: {val:6.2f} um")