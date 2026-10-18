"""
vendor_thickness_confound_check.py
====================================
R26-DS-015 -- OCT/MS branch, standalone check (NOT wired into the
dashboard). Answers one question cheaply before committing to building a
full vol-only segmentation model: does the ONE number Spectralis's own
built-in auto-segmentation actually gives you (total retinal thickness,
confirmed by direct inspection of your real .vol files to be the only
reliably populated signal -- see chat) separate MS from HC at all on your
own 35 subjects?

This deliberately uses ONLY the .vol files -- no .mat, no manual
delineation. That's the whole point: it's a dry run of the real-world
"vol-only" scenario (a clinic hands you a raw scan, nothing else), using
data where you happen to also know the true label, so you can check the
result honestly.

How the vendor boundaries are found: Spectralis .vol files allocate a
fixed number of boundary "slots" (num_seg, varies per file -- confirmed
3 slots in one of your HC subjects, 10 in one of your MS subjects), but
in both files checked, only 2 of those slots actually contained real
(non-NaN) data -- the rest were empty placeholders. This script doesn't
assume a fixed boundary count or index; for each subject it finds
whichever boundary slots are actually populated (>50% non-NaN across the
scan) and uses the topmost and bottommost of those as the top and bottom
of "total retinal thickness." If a subject has fewer than 2 populated
boundaries, it's skipped and reported, not silently dropped.

Same LOOCV methodology as every other comparison in this project
(build_multifeature_classifier.py's confound checks, the OCT thickness
model itself): honest small-n evaluation, scaler fit only on the
training fold each time.

Usage:
    python3 vendor_thickness_confound_check.py \\
        "/Users/tanuridissanayaka/Research Datasets/MS OCT/OCT_Manual_Delineations-2018_June_29" \\
        --out vendor_thickness_results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from vol_loader import load_vol
from thickness import central_bscan_window

VALID_FRACTION_THRESHOLD = 0.5  # a boundary slot counts as "real" if >50% of its values are non-NaN


def _find_populated_boundaries(vendor_layers: np.ndarray) -> list:
    """vendor_layers: (n_bscans, num_seg, size_x). Returns indices of slots
    that are actually populated with real data, sorted top-to-bottom by
    mean y-value (so this doesn't assume storage order == anatomical order)."""
    n_bscans, num_seg, size_x = vendor_layers.shape
    populated = []
    for i in range(num_seg):
        frac_valid = np.mean(~np.isnan(vendor_layers[:, i, :]))
        if frac_valid > VALID_FRACTION_THRESHOLD:
            mean_y = np.nanmean(vendor_layers[:, i, :])
            populated.append((i, mean_y))
    populated.sort(key=lambda t: t[1])  # smaller y = shallower = more anatomically "top"
    return [i for i, _ in populated]


def compute_total_thickness(root: str, out_csv: str, central_window: int = 10) -> pd.DataFrame:
    root_path = Path(root)
    vol_dir = root_path / "vol"
    vol_files = sorted(vol_dir.glob("*.vol"))
    print(f"Found {len(vol_files)} .vol files\n")

    rows = []
    failed = []

    for vol_path in vol_files:
        subject_id = vol_path.stem.split("_")[0]
        try:
            vol = load_vol(str(vol_path))
            populated = _find_populated_boundaries(vol.vendor_layers)
            if len(populated) < 2:
                raise ValueError(f"only {len(populated)} populated boundary slot(s) found "
                                  f"(need >= 2 for a top/bottom thickness) -- "
                                  f"slots checked: {vol.vendor_layers.shape[1]}")

            top_idx, bottom_idx = populated[0], populated[-1]
            top = vol.vendor_layers[:, top_idx, :]
            bottom = vol.vendor_layers[:, bottom_idx, :]

            thickness_px = bottom - top  # (n_bscans, size_x)
            thickness_px = np.where(thickness_px < 0, np.nan, thickness_px)
            scale_y_um = vol.meta["scale_y"] * 1000.0
            thickness_um = thickness_px * scale_y_um

            per_scan_um = np.nanmean(thickness_um, axis=1)  # (n_bscans,)
            window = central_bscan_window(vol.meta["n_bscans"], window=central_window)
            central_total_um = float(np.nanmean(per_scan_um[window]))
            whole_total_um = float(np.nanmean(per_scan_um))

        except Exception as e:
            failed.append(f"[{subject_id}] {e}")
            continue

        label = 0 if subject_id.startswith("hc") else 1
        rows.append({
            "subject_id": subject_id,
            "label": label,
            "group": "Healthy" if label == 0 else "MS",
            "n_populated_boundaries": len(populated),
            "total_thickness_whole_um": whole_total_um,
            "total_thickness_central_um": central_total_um,
        })
        print(f"  {subject_id:6s} ({rows[-1]['group']:7s}) -- "
              f"{len(populated)} populated boundaries, total thickness (central) = {central_total_um:.1f} um")

    df = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f"\nProcessed {len(df)}/{len(vol_files)} subjects successfully.")
    if failed:
        print(f"\n⚠️  {len(failed)} subjects FAILED (not in output):")
        for f in failed:
            print(f"  {f}")

    return df


def run_loocv(X: np.ndarray, y: np.ndarray, model_fn) -> dict:
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = model_fn()
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)[0, 1]
        y_true.append(y[test_idx][0]); y_pred.append(pred); y_proba.append(proba)
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    return {"accuracy": accuracy_score(y_true, y_pred), "roc_auc": roc_auc_score(y_true, y_proba)}


def main(root: str, out_csv: str, central_window: int = 10) -> None:
    df = compute_total_thickness(root, out_csv, central_window=central_window)

    n_hc = (df["label"] == 0).sum()
    n_ms = (df["label"] == 1).sum()
    majority_baseline = max(n_hc, n_ms) / len(df)

    print("\n" + "=" * 80)
    print("CONFOUND CHECK: does total retinal thickness ALONE (the one real number")
    print("vendor auto-segmentation gives you) separate MS from HC at all?")
    print("=" * 80)
    print(f"(Majority-class baseline, i.e. always guessing the bigger group: {majority_baseline:.3f} "
          f"-- {n_hc} HC, {n_ms} MS)\n")

    print("Group means (central-window total thickness, microns):")
    print(f"  HC: {df[df.label==0]['total_thickness_central_um'].mean():.1f} um  "
          f"(SD {df[df.label==0]['total_thickness_central_um'].std():.1f})")
    print(f"  MS: {df[df.label==1]['total_thickness_central_um'].mean():.1f} um  "
          f"(SD {df[df.label==1]['total_thickness_central_um'].std():.1f})")

    X = df[["total_thickness_central_um"]].values
    y = df["label"].values
    lr = run_loocv(X, y, lambda: LogisticRegression(max_iter=1000))
    rf = run_loocv(X, y, lambda: RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42))
    print(f"\nLOOCV, total thickness alone:")
    print(f"  Logistic Regression: acc={lr['accuracy']:.3f}  auc={lr['roc_auc']:.3f}")
    print(f"  Random Forest:       acc={rf['accuracy']:.3f}  auc={rf['roc_auc']:.3f}")

    print("\nInterpretation:")
    best_acc = max(lr["accuracy"], rf["accuracy"])
    best_auc = max(lr["roc_auc"], rf["roc_auc"])
    if best_acc <= majority_baseline + 0.03 and best_auc < 0.60:
        print(f"  Total thickness alone does NOT meaningfully separate MS from HC here --")
        print(f"  accuracy is at/near the majority-class baseline ({majority_baseline:.3f}) and AUC")
        print(f"  ({best_auc:.3f}) is close to chance (0.5). This confirms a vol-only shortcut using")
        print(f"  just the vendor auto-segmentation's single thickness number would not work even")
        print(f"  in principle -- the full 9-boundary (or at minimum 4-band) per-layer detail is")
        print(f"  doing the real work, not overall retinal thickness.")
    else:
        print(f"  Total thickness alone shows SOME separation above the majority-class baseline")
        print(f"  ({majority_baseline:.3f}) and chance AUC -- worth a second look before ruling out")
        print(f"  a vol-only approach, though still almost certainly weaker than per-layer features.")

    print(f"\nSaved per-subject results to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to OCT_Manual_Delineations-2018_June_29 folder (only vol/ is used)")
    parser.add_argument("--out", default="vendor_thickness_results.csv")
    parser.add_argument("--central-window", type=int, default=10)
    args = parser.parse_args()
    main(args.dataset_root, args.out, central_window=args.central_window)
