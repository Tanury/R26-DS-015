"""
recompute_4band_features.py
============================
R26-DS-015 -- eye/ms preprocessing / external validation prep

Recomputes a coarser, 4-band retinal thickness feature set from the
EXISTING 35 subjects' already-parsed 9-boundary manual delineations, so
that this subset of the data can later be matched against OCT5k's
coarser 5-boundary / 4-band annotation scheme (OCT5k annotates AMD/DME/
healthy OCT volumes, not MS -- see run_oct5k_specificity.py, pending the
actual OCT5k download).

This does NOT re-delineate anything or add new data -- it is a pure
re-aggregation of the SAME 9 boundaries already confirmed in
mat_loader.py, selecting the 5 of them that correspond to OCT5k's 5
annotated boundaries and discarding the rest. No external data is needed
for this step.

Boundary correspondence (confirmed against the OCT5k paper's own
guideline text for its annotators, fetched 2026-10-15 -- see note below
each one):

    OCT5k boundary   mat_loader.py boundary (index)   how it was confirmed
    --------------   -------------------------------  ---------------------
    ILM              ILM            (0)               exact name match
    OPL              INL-OPL        (3)               OCT5k guideline: "OPL
                                                        is segmented on top
                                                        of the bright band of
                                                        the OPL" -- i.e. the
                                                        boundary is the TOP
                                                        edge of the OPL band,
                                                        which is the INL/OPL
                                                        transition, not
                                                        OPL/ONL.
    IS-OS            IS-OS          (6)               exact name match
    IBRPE            OS-RPE         (7)               OCT5k: "Inner boundary
                                                        of RPE" = top/inner
                                                        edge of the RPE
                                                        complex = the OS/RPE
                                                        transition.
    OBRPE            BM             (8)               OCT5k: "outermost
                                                        highly reflective
                                                        layer... traced along
                                                        the bottom band" =
                                                        Bruch's membrane, the
                                                        bottom of the RPE
                                                        complex.

So the 5 boundaries kept, by index into the original 9-boundary array
(0 ILM, 1 RNFL-GCL, 2 IPL-INL, 3 INL-OPL, 4 OPL-ONL, 5 ELM, 6 IS-OS,
7 OS-RPE, 8 BM), are: [0, 3, 6, 7, 8].

compute_thickness_features() is generic over boundary count (it just
takes consecutive differences), so feeding it this 5-boundary slice
directly gives the 4 collapsed bands with zero reimplementation risk --
same masking/averaging logic as the existing 8-layer pipeline.

The 4 resulting bands and what each absorbs from the original 8 layers:

    Band              Absorbs (original LAYER_NAMES)
    ----              -------------------------------
    ILM_OPL           RNFL + GCIP + INL
    OPL_ISOS          OPL + ONL + IS
    ISOS_IBRPE        OS
    IBRPE_OBRPE       RPE

IMPORTANT CAVEAT (read before trusting downstream results): this mapping
is based on the OCT5k paper's written annotator guidelines, not on a
visual check against an actual OCT5k sample image/mask. Boundary-naming
conventions have genuine inconsistency across the retinal-OCT literature
(e.g. some papers put "ELM" where others don't annotate it at all, and
"OPL" has been used for either edge of the band in different papers), so
once real OCT5k sample files are in hand, sanity-check this mapping by
overlaying the 4 recomputed boundaries from one of THIS dataset's known
subjects against the visual B-scan (oct_visualization.py already does
this) and confirming the bands look anatomically sensible before trusting
any specificity number computed downstream.

Usage:
    python3 recompute_4band_features.py \\
        "/Users/tanuridissanayaka/Research Datasets/MS OCT/OCT_Manual_Delineations-2018_June_29" \\
        --out oct5k_4band_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from vol_loader import load_vol
from mat_loader import load_boundaries
from thickness import compute_thickness_features, central_bscan_window

# Index into the original 9-boundary array (see mapping table above).
KEPT_BOUNDARY_IDX = [0, 3, 6, 7, 8]
BAND_NAMES = ["ILM_OPL", "OPL_ISOS", "ISOS_IBRPE", "IBRPE_OBRPE"]


def recompute_4band(root: str, out_csv: str, central_window: int = 10) -> pd.DataFrame:
    root_path = Path(root)
    vol_dir = root_path / "vol"
    mat_dir = root_path / "delineation"

    vol_files = sorted(vol_dir.glob("*.vol"))
    print(f"Found {len(vol_files)} .vol files\n")

    rows = []
    failed = []

    for vol_path in vol_files:
        subject_id = vol_path.stem
        mat_path = mat_dir / f"{subject_id}.mat"

        try:
            vol = load_vol(str(vol_path))
            bd = load_boundaries(str(mat_path), size_x=vol.meta["size_x"])

            if bd.n_boundaries != 9:
                raise ValueError(
                    f"expected 9 boundaries, got {bd.n_boundaries} -- "
                    f"KEPT_BOUNDARY_IDX assumes the standard 9-boundary layout"
                )

            boundaries_4band = bd.boundaries[:, KEPT_BOUNDARY_IDX, :]  # (n_bscans, 5, size_x)

            feats_whole = compute_thickness_features(boundaries_4band, scale_y_mm=vol.meta["scale_y"])
            feats_whole.layer_names = list(BAND_NAMES)

            window = central_bscan_window(vol.meta["n_bscans"], window=central_window)
            feats_central = compute_thickness_features(
                boundaries_4band, scale_y_mm=vol.meta["scale_y"], bscan_indices=window
            )
            feats_central.layer_names = list(BAND_NAMES)

        except Exception as e:
            failed.append(f"[{subject_id}] {e}")
            continue

        short_id = subject_id.split("_")[0]
        label = 0 if short_id.startswith("hc") else 1

        row = {
            "subject_id": short_id,
            "label": label,
            "group": "Healthy" if label == 0 else "MS",
            "mat_format": bd.source_format,
        }
        for name, val in zip(BAND_NAMES, feats_whole.subject_vector):
            row[f"band_{name}_um"] = val
        for name, val in zip(BAND_NAMES, feats_central.subject_vector):
            row[f"band_{name}_central_um"] = val
        rows.append(row)

        print(f"  {short_id:6s} ({row['group']:7s}) -- whole: "
              f"{', '.join(f'{n}={v:.1f}' for n, v in zip(BAND_NAMES, feats_whole.subject_vector))}")

    df = pd.DataFrame(rows)
    df = df.sort_values("subject_id").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f"\nProcessed {len(df)}/{len(vol_files)} subjects successfully.")
    print(f"  Healthy: {(df['label'] == 0).sum()}   MS: {(df['label'] == 1).sum()}")
    print(f"Saved to: {out_csv}")

    if failed:
        print(f"\n⚠️  {len(failed)} subjects FAILED (not in output):")
        for f in failed:
            print(f"  {f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to OCT_Manual_Delineations-2018_June_29 folder")
    parser.add_argument("--out", default="oct5k_4band_features.csv")
    parser.add_argument("--central-window", type=int, default=10)
    args = parser.parse_args()
    recompute_4band(args.dataset_root, args.out, central_window=args.central_window)
