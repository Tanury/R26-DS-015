"""
validate_oct_data.py
=====================
eye/ms preprocessing

Data-quality validation for the OCT branch, on top of 
check_dataset_consistency.py. That script checks GEOMETRY consistency
(sizes, scan counts). This script checks whether the actual parsed
VALUES are trustworthy:

  1. Boundary-crossing rate per subject -- compute_thickness_features()
     already masks negative (crossed) boundary differences as NaN
     silently. This re-surfaces that masking as a per-subject count:
     a subject with a high crossing rate likely has a parsing or
     column-order bug, not just noisy manual delineation.

  2. Outlier check on subject-level thickness features -- flags any
     subject whose value for a layer is > 3 SD from that LAYER's
     HC-only mean (data-driven, since exact literature RNFL/GCIP
     normative ranges vary by device/population and shouldn't be
     hardcoded as ground truth here). Flagged subjects are candidates
     for a manual re-check of their .mat file, not automatic removal.

  3. Duplicate-subject / fellow-eye leakage check -- if your dataset's
     filenames encode left/right eye (commonly a trailing _R / _L or
     similar) for the SAME underlying person, treating both eyes as
     independent samples in LOOCV overstates n and can leak subject
     identity across folds. This flags any subject_id base (before an
     eye marker) that appears more than once.

Usage:
    python3 validate_oct_data.py \\
        "/Users/tanuridissanayaka/Research Datasets/MS OCT/OCT_Manual_Delineations-2018_June_29" \\
        thickness_features.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from vol_loader import load_vol
from mat_loader import load_boundaries
from thickness import LAYER_NAMES

# Matches a trailing eye marker so "hc01_..._R" and "hc01_..._L" are
# recognized as the SAME underlying subject. Extend this if your
# filenames use a different convention (check a few names first).
EYE_SUFFIX_RE = re.compile(r"^(?P<base>[a-zA-Z]+\d+)")


def check_boundary_crossings(root: str) -> pd.DataFrame:
    root_path = Path(root)
    vol_dir = root_path / "vol"
    mat_dir = root_path / "delineation"

    rows = []
    for vol_path in sorted(vol_dir.glob("*.vol")):
        subject_id = vol_path.stem
        mat_path = mat_dir / f"{subject_id}.mat"
        if not mat_path.exists():
            continue
        try:
            vol = load_vol(str(vol_path))
            bd = load_boundaries(str(mat_path), size_x=vol.meta["size_x"])
        except Exception as e:
            rows.append({"subject_id": subject_id, "error": str(e)})
            continue

        # Recompute the raw diff BEFORE compute_thickness_features masks it,
        # so we can report how often it happened instead of just hiding it.
        thickness_px = np.diff(bd.boundaries, axis=1)  # (n_bscans, n_layers, size_x)
        n_total = thickness_px.size
        n_crossed = int(np.sum(thickness_px < 0))
        rows.append({
            "subject_id": subject_id,
            "n_boundary_points": n_total,
            "n_crossings": n_crossed,
            "crossing_rate_pct": round(100 * n_crossed / n_total, 3) if n_total else None,
        })

    return pd.DataFrame(rows)


def check_outliers(csv_path: str, z_thresh: float = 3.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    flagged = []
    for layer in LAYER_NAMES:
        col = f"thickness_{layer}_um"
        if col not in df.columns:
            continue
        hc_vals = df.loc[df["label"] == 0, col]
        mean, sd = hc_vals.mean(), hc_vals.std()
        if sd == 0 or np.isnan(sd):
            continue
        z = (df[col] - mean) / sd
        for idx in df.index[z.abs() > z_thresh]:
            flagged.append({
                "subject_id": df.loc[idx, "subject_id"],
                "group": df.loc[idx, "group"],
                "layer": layer,
                "value_um": round(df.loc[idx, col], 2),
                "hc_mean_um": round(mean, 2),
                "z_score": round(z[idx], 2),
            })
    return pd.DataFrame(flagged)


def check_eye_duplicates(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    bases = df["subject_id"].apply(lambda s: EYE_SUFFIX_RE.match(s).group("base")
                                    if EYE_SUFFIX_RE.match(s) else s)
    counts = bases.value_counts()
    dupes = counts[counts > 1]
    if dupes.empty:
        return pd.DataFrame(columns=["base_subject", "count", "subject_ids"])
    rows = []
    for base, n in dupes.items():
        ids = df.loc[bases == base, "subject_id"].tolist()
        rows.append({"base_subject": base, "count": n, "subject_ids": ", ".join(ids)})
    return pd.DataFrame(rows)


def main(dataset_root: str, thickness_csv: str) -> None:
    print("=" * 70)
    print("1. BOUNDARY-CROSSING CHECK (parsing/annotation sanity)")
    print("=" * 70)
    crossings = check_boundary_crossings(dataset_root)
    if "error" in crossings.columns and crossings["error"].notna().any():
        print("  Subjects that failed to parse:")
        print(crossings[crossings["error"].notna()][["subject_id", "error"]].to_string(index=False))
    ok = crossings[crossings.get("error").isna()] if "error" in crossings.columns else crossings
    if not ok.empty:
        worst = ok.sort_values("crossing_rate_pct", ascending=False).head(5)
        print(f"\n  Mean crossing rate across {len(ok)} subjects: "
              f"{ok['crossing_rate_pct'].mean():.3f}%")
        print("  5 highest crossing-rate subjects (investigate if notably above the mean):")
        print(worst.to_string(index=False))

    print("\n" + "=" * 70)
    print("2. OUTLIER CHECK (subject thickness vs HC-group mean, |z| > 3)")
    print("=" * 70)
    outliers = check_outliers(thickness_csv)
    if outliers.empty:
        print("  No subjects flagged -- no layer value is > 3 SD from the HC mean.")
    else:
        print(outliers.to_string(index=False))
        print("\n  NOTE: a flagged MS subject may be a true, large biological effect "
              "(that's the point of the biomarker). A flagged HC subject is more "
              "suspicious and worth a manual boundary re-check.")

    print("\n" + "=" * 70)
    print("3. FELLOW-EYE / DUPLICATE-SUBJECT CHECK")
    print("=" * 70)
    dupes = check_eye_duplicates(thickness_csv)
    if dupes.empty:
        print("  No repeated base subject IDs found -- each subject contributes once.")
    else:
        print(dupes.to_string(index=False))
        print("\n  If these are the same person's left/right eye, decide explicitly: "
              "keep one eye per subject (recommended, keeps LOOCV folds independent), "
              "or average the two before featurizing. Do NOT let both count as two "
              "independent LOOCV samples.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])