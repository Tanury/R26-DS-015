"""
mat_loader.py
=============
R26-DS-015 — eye/ms preprocessing

Loads the manual retinal-layer delineation .mat files that accompany the
JHU OCT dataset. This dataset was released in two zip variants with two
different .mat structures (confirmed against real samples from both):

  Format A -- 'control_pts' cell array (32/35 subjects in this download):
    control_pts: (n_bscans, 11) object array
        - 9 of the 11 columns are populated per B-scan (the rest are
          empty placeholders) — 9 boundaries define 8 retinal layers.
        - Each populated cell is (21, 2): sparse (x, y) spline control
          points along that boundary, in pixel coordinates matching the
          corresponding .vol B-scan (x in [1, size_x], y in [0, size_y]).
    segmode: scalar flag, not used here.
    Densified via monotonic cubic (PCHIP) interpolation of the sparse points.

  Format B -- 'bd_pts' dense array (ms14, ms16, ms17 in this download --
  apparently from the "_b" zip variant of this dataset):
    bd_pts: (size_x, n_bscans, 9) float64
        - Already dense per-pixel boundary y-values, no interpolation
          needed -- just transpose to this module's (n_bscans, n_boundaries,
          size_x) convention.

Both formats are handled transparently by load_boundaries(); this module
is a fresh implementation for R26-DS-015, not derived from any
classification repository.

Confirmed boundary order (9 boundaries; matches He et al., Data in Brief
22:601-604, 2019 -- the GCL-IPL boundary is intentionally skipped since
it isn't visible on OCT, which is why one column is always empty in
Format A):

    0 ILM   1 RNFL-GCL   2 IPL-INL   3 INL-OPL   4 OPL-ONL
    5 ELM   6 IS-OS      7 OS-RPE    8 BM

The 8 resulting layers (thickness.py computes these as consecutive
boundary differences):

    0 RNFL   1 GCIP   2 INL   3 OPL   4 ONL   5 IS   6 OS   7 RPE

Usage:
    from mat_loader import load_boundaries
    boundaries = load_boundaries("hc01_spectralis_macula_v1_s1_R.mat", size_x=1024)
    boundaries.shape  # (n_bscans, n_boundaries, size_x) -- dense y-values, nan where undelineated
"""

from dataclasses import dataclass

import numpy as np
import scipy.io as sio
from scipy.interpolate import PchipInterpolator

# Confirmed against He et al. 2019 (Data in Brief) and cross-checked against
# the always-empty columns found in this dataset's Format-A .mat files.
BOUNDARY_NAMES = [
    "ILM", "RNFL-GCL", "IPL-INL", "INL-OPL",
    "OPL-ONL", "ELM", "IS-OS", "OS-RPE", "BM",
]
LAYER_NAMES = ["RNFL", "GCIP", "INL", "OPL", "ONL", "IS", "OS", "RPE"]


@dataclass
class BoundaryData:
    boundaries: np.ndarray  # (n_bscans, n_boundaries, size_x) float, nan = missing
    n_bscans: int
    n_boundaries: int
    source_format: str  # "control_pts" or "bd_pts", for provenance/debugging


def _populated_columns(control_pts: np.ndarray) -> list[int]:
    """Find which of the 11 cell-array columns actually contain data
    (consistently empty columns across all scans are unused placeholders)."""
    n_bscans, n_cols = control_pts.shape
    populated = []
    for j in range(n_cols):
        if any(control_pts[i, j].shape != (0, 0) for i in range(n_bscans)):
            populated.append(j)
    return populated


def _load_from_control_pts(control_pts: np.ndarray, size_x: int) -> BoundaryData:
    n_bscans, _ = control_pts.shape
    boundary_cols = _populated_columns(control_pts)
    n_boundaries = len(boundary_cols)

    dense = np.full((n_bscans, n_boundaries, size_x), np.nan, dtype=np.float64)
    x_grid = np.arange(1, size_x + 1)

    for i in range(n_bscans):
        for out_b, col in enumerate(boundary_cols):
            pts = control_pts[i, col]
            if pts.shape == (0, 0):
                continue  # this scan has no delineation for this boundary
            xs, ys = pts[:, 0], pts[:, 1]
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            # dedupe identical x (PCHIP requires strictly increasing x)
            keep = np.concatenate([[True], np.diff(xs) > 0])
            xs, ys = xs[keep], ys[keep]
            if len(xs) < 2:
                continue
            interp = PchipInterpolator(xs, ys, extrapolate=False)
            dense[i, out_b, :] = interp(x_grid)

    return BoundaryData(
        boundaries=dense, n_bscans=n_bscans, n_boundaries=n_boundaries,
        source_format="control_pts",
    )


def _load_from_bd_pts(bd_pts: np.ndarray, size_x: int) -> BoundaryData:
    # bd_pts: (size_x_file, n_bscans, n_boundaries) -> (n_bscans, n_boundaries, size_x)
    if bd_pts.shape[0] != size_x:
        raise ValueError(
            f"bd_pts first dim ({bd_pts.shape[0]}) doesn't match expected size_x "
            f"({size_x}) — check the paired .vol file's size_x before proceeding."
        )
    dense = np.transpose(bd_pts, (1, 2, 0)).astype(np.float64)  # (n_bscans, n_boundaries, size_x)
    n_bscans, n_boundaries, _ = dense.shape
    return BoundaryData(
        boundaries=dense, n_bscans=n_bscans, n_boundaries=n_boundaries,
        source_format="bd_pts",
    )


def load_boundaries(path: str, size_x: int = 1024) -> BoundaryData:
    """Load manual delineation boundaries, auto-detecting which of the two
    known .mat formats this file uses.

    Args:
        path:   path to the .mat file
        size_x: B-scan width in pixels (must match the paired .vol file's
                meta["size_x"])
    """
    data = sio.loadmat(path)

    if "control_pts" in data:
        return _load_from_control_pts(data["control_pts"], size_x)
    elif "bd_pts" in data:
        return _load_from_bd_pts(data["bd_pts"], size_x)
    else:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(
            f"Neither 'control_pts' nor 'bd_pts' found in {path}. "
            f"Actual top-level keys: {keys} -- this file uses a third, "
            f"unhandled format and needs its own inspection."
        )


if __name__ == "__main__":
    import sys

    bd = load_boundaries(sys.argv[1], size_x=int(sys.argv[2]) if len(sys.argv) > 2 else 1024)
    print(f"source format: {bd.source_format}")
    print(f"boundaries shape: {bd.boundaries.shape}")
    print(f"nan fraction: {np.isnan(bd.boundaries).mean():.3f}")
    print(f"boundary 0, scan 0, first/last valid y: "
          f"{bd.boundaries[0, 0][~np.isnan(bd.boundaries[0, 0])][[0, -1]]}")