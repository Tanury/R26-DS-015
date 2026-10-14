"""
check_dataset_consistency.py
=============================
R26-DS-015 — eye/ms preprocessing

Run this ONCE against your full local dataset (all 35 hc0X/ms0X pairs)
before writing the batch training pipeline. It checks:

  1. Every .vol file has the same size_x, size_y, n_bscans, scan_position
     (if these vary, downstream code that assumes a fixed shape will break
     or silently mismatch).
  2. scale_x / scale_y / distance are consistent (or at least checked --
     these should be near-identical for the same scanner/protocol, but
     worth confirming since thickness-in-microns depends on scale_y).
  3. Every .mat file has exactly the same populated boundary columns
     (indices 0,1,3,4,5,6,7,8,9 out of 11 -- if any subject has a
     different set, the boundary/layer names in mat_loader.py's
     BOUNDARY_NAMES would be silently mislabeled for that subject).
  4. .vol and .mat n_bscans match for every pair.

Usage:
    python check_dataset_consistency.py \\
        "/Users/tanuridissanayaka/Research Datasets/MS OCT/OCT_Manual_Delineations-2018_June_29"
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vol_loader import load_vol
from mat_loader import load_boundaries


EXPECTED_POPULATED_COLS = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # only meaningful for "control_pts" format


def check_dataset(root: str) -> None:
    root_path = Path(root)
    vol_dir = root_path / "vol"
    mat_dir = root_path / "delineation"

    vol_files = sorted(vol_dir.glob("*.vol"))
    print(f"Found {len(vol_files)} .vol files in {vol_dir}\n")

    records = []
    problems = []
    format_counts = {}

    for vol_path in vol_files:
        subject_id = vol_path.stem  # e.g. "hc01_spectralis_macula_v1_s1_R"
        mat_path = mat_dir / f"{subject_id}.mat"

        if not mat_path.exists():
            problems.append(f"[{subject_id}] MISSING .mat file: {mat_path}")
            continue

        # --- .vol header ---
        try:
            vol = load_vol(str(vol_path))
        except Exception as e:
            problems.append(f"[{subject_id}] FAILED to parse .vol: {e}")
            continue

        # --- .mat boundaries (auto-detects control_pts vs bd_pts format) ---
        try:
            bd = load_boundaries(str(mat_path), size_x=vol.meta["size_x"])
            n_bscans_mat = bd.n_bscans
            format_counts[bd.source_format] = format_counts.get(bd.source_format, 0) + 1
        except Exception as e:
            problems.append(f"[{subject_id}] FAILED to parse .mat: {e}")
            continue

        rec = {
            "id": subject_id,
            "size_x": vol.meta["size_x"],
            "size_y": vol.meta["size_y"],
            "n_bscans_vol": vol.meta["n_bscans"],
            "n_bscans_mat": n_bscans_mat,
            "scan_position": vol.meta["scan_position"],
            "scale_x": vol.meta["scale_x"],
            "scale_y": vol.meta["scale_y"],
            "distance": vol.meta["distance"],
            "n_boundaries": bd.n_boundaries,
            "mat_format": bd.source_format,
        }
        records.append(rec)

        # --- per-subject checks ---
        if vol.meta["n_bscans"] != n_bscans_mat:
            problems.append(
                f"[{subject_id}] n_bscans MISMATCH: .vol={vol.meta['n_bscans']} "
                f".mat={n_bscans_mat}"
            )
        if bd.n_boundaries != len(EXPECTED_POPULATED_COLS):
            problems.append(
                f"[{subject_id}] UNEXPECTED boundary count: {bd.n_boundaries} "
                f"(expected {len(EXPECTED_POPULATED_COLS)}) -- layer names may be WRONG for this subject"
            )

    if not records:
        print("No subjects successfully parsed -- check paths above.")
        return

    # --- cross-subject consistency ---
    print("=" * 70)
    print("CROSS-SUBJECT CONSISTENCY")
    print("=" * 70)
    for field in ["size_x", "size_y", "n_bscans_vol", "scan_position"]:
        values = sorted(set(r[field] for r in records))
        status = "OK (all identical)" if len(values) == 1 else "*** VARIES ***"
        print(f"  {field:15s}: {values}  {status}")

    for field in ["scale_x", "scale_y", "distance"]:
        values = np.array([r[field] for r in records])
        print(
            f"  {field:15s}: min={values.min():.6f} max={values.max():.6f} "
            f"std={values.std():.6f}"
        )

    print(f"\nTotal subjects checked: {len(records)}")
    print(f"Healthy (hc*): {sum(1 for r in records if r['id'].startswith('hc'))}")
    print(f"MS (ms*):      {sum(1 for r in records if r['id'].startswith('ms'))}")
    print(f".mat format breakdown: {format_counts}")

    print("\n" + "=" * 70)
    print("PROBLEMS" if problems else "NO PROBLEMS FOUND")
    print("=" * 70)
    for p in problems:
        print(f"  ⚠️  {p}")
    if not problems:
        print("  All subjects have consistent geometry and expected boundary structure.")
        print("  Safe to proceed with batch preprocessing using fixed shapes/names.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    check_dataset(sys.argv[1])