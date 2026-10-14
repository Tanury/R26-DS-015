"""
compare_ms3seg_lemon_features.py
==================================
MRI-MS branch (MindGlide-based)

Compares MindGlide's OWN lesion volumes between MS3SEG patients and
LEMON healthy controls -- this is the comparison that actually matters
for classifier feasibility, since at real inference time we only ever
have MindGlide's predicted volume, never expert ground truth. (Note:
this is different from comparing MindGlide's volumes against MS3SEG's
ground-truth masks, which is what validate_mindglide_on_ms3seg.py did.)

Also runs a simple single-feature LOOCV separability check (logistic
regression on lesion_volume_mm3 alone) so "can this one feature
separate the groups" is answered with a number, not by eyeballing
min/max ranges.

Usage:
    python3 compare_ms3seg_lemon_features.py \\
        mindglide_validation.csv mindglide_lemon_hc.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


def load_ms3seg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str})
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[["patient_id", "mindglide_lesion_volume_mm3"]].copy()
    df.columns = ["id", "lesion_volume_mm3"]
    df["label"] = 1  # MS
    return df


def load_lemon(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"subject_id": str})
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[["subject_id", "lesion_volume_mm3"]].copy()
    df.columns = ["id", "lesion_volume_mm3"]
    df["label"] = 0  # Healthy
    return df


def run_loocv(X: np.ndarray, y: np.ndarray) -> dict:
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = LogisticRegression()
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)[0, 1]
        y_true.append(y[test_idx][0]); y_pred.append(pred); y_proba.append(proba)
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "y_true": y_true, "y_pred": y_pred,
    }


def main(ms3seg_csv: str, lemon_csv: str) -> None:
    ms = load_ms3seg(ms3seg_csv)
    hc = load_lemon(lemon_csv)
    combined = pd.concat([ms, hc], ignore_index=True)

    print("=" * 65)
    print("MindGlide's OWN lesion volumes -- MS3SEG (MS) vs LEMON (Healthy)")
    print("=" * 65)
    print(f"\nMS3SEG (n={len(ms)}):")
    print(f"  mean={ms['lesion_volume_mm3'].mean():.0f}  std={ms['lesion_volume_mm3'].std():.0f}  "
          f"min={ms['lesion_volume_mm3'].min():.0f}  max={ms['lesion_volume_mm3'].max():.0f}")
    print(f"\nLEMON (n={len(hc)}):")
    print(f"  mean={hc['lesion_volume_mm3'].mean():.0f}  std={hc['lesion_volume_mm3'].std():.0f}  "
          f"min={hc['lesion_volume_mm3'].min():.0f}  max={hc['lesion_volume_mm3'].max():.0f}")

    overlap_count = ((hc["lesion_volume_mm3"].max() >= ms["lesion_volume_mm3"].min())
                      if len(ms) and len(hc) else False)
    n_ms_below_hc_max = (ms["lesion_volume_mm3"] <= hc["lesion_volume_mm3"].max()).sum()
    n_hc_above_ms_min = (hc["lesion_volume_mm3"] >= ms["lesion_volume_mm3"].min()).sum()
    print(f"\nOverlap check:")
    print(f"  MS3SEG patients at/below LEMON's max ({hc['lesion_volume_mm3'].max():.0f} mm³): "
          f"{n_ms_below_hc_max}/{len(ms)}")
    print(f"  LEMON subjects at/above MS3SEG's min ({ms['lesion_volume_mm3'].min():.0f} mm³): "
          f"{n_hc_above_ms_min}/{len(hc)}")

    print("\n" + "=" * 65)
    print("SINGLE-FEATURE SEPARABILITY (LOOCV, lesion_volume_mm3 alone)")
    print("=" * 65)
    X = combined[["lesion_volume_mm3"]].values
    y = combined["label"].values
    results = run_loocv(X, y)
    print(f"Accuracy: {results['accuracy']:.3f}  |  ROC-AUC: {results['roc_auc']:.3f}")
    misclassified = combined.loc[results["y_true"] != results["y_pred"], "id"].tolist()
    print(f"Misclassified: {misclassified if misclassified else 'None'}")

    print("\nInterpretation:")
    print("  ROC-AUC > 0.85  -- lesion volume alone is a strong feature, worth building on")
    print("  ROC-AUC 0.7-0.85 -- moderate; usable but expect other features/modalities to matter too")
    print("  ROC-AUC < 0.7   -- lesion volume alone is not reliably separating these groups on")
    print("                     this small sample -- investigate before committing further")
    print("\n(Small-n caveat: with only "
          f"{len(ms)} MS3SEG + {len(hc)} LEMON subjects, these numbers will shift as you")
    print("process more of both cohorts -- treat this as an early read, not a final result.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ms3seg_csv", help="Path to mindglide_validation.csv (from validate_mindglide_on_ms3seg.py)")
    parser.add_argument("lemon_csv", help="Path to mindglide_lemon_hc.csv (from check_mindglide_on_lemon_hc.py)")
    args = parser.parse_args()
    main(args.ms3seg_csv, args.lemon_csv)