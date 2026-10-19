"""
OCT/MS branch, standalone validation script (NOT wired into
the live dashboard -- scope deliberately limited to this, per project
decision: a 9-boundary/8-layer model stays the only model the dashboard
serves; this comparison-only model exists purely to let OCT5k's AMD/DME
subjects be run through SOMETHING comparable, for a specificity check).

Trains a small classifier on the 4-band features produced by
recompute_4band_features.py (the SAME subjects, 9 boundaries collapsed to the 5), 

"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

BAND_NAMES = ["ILM_OPL", "OPL_ISOS", "ISOS_IBRPE", "IBRPE_OBRPE"]

# 65.7% / AUC from the full 8-layer model's own LOOCV (already established
# in this project, from the Wilson-CI accuracy-ceiling discussion). This is
# the number the 4-band version is being compared against, not a new claim.
FULL_MODEL_LOOCV_ACCURACY = 0.657


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
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "y_true": y_true, "y_pred": y_pred,
    }


def report(name: str, X: np.ndarray, y: np.ndarray, model_fn) -> dict:
    result = run_loocv(X, y, model_fn)
    print(f"{name:<40s} acc={result['accuracy']:.3f}  auc={result['roc_auc']:.3f}")
    return result


def main(csv_path: str, out_model: str = "4band_comparison_model.joblib") -> None:
    df = pd.read_csv(csv_path, dtype={"subject_id": str})
    print(f"Loaded {len(df)} subjects: {(df['label']==0).sum()} Healthy, {(df['label']==1).sum()} MS\n")

    feature_cols = [f"band_{name}_central_um" for name in BAND_NAMES]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found: {missing} -- did you run "
                        f"recompute_4band_features.py first?")

    X = df[feature_cols].values
    y = df["label"].values

    print("=" * 80)
    print("LOOCV comparison: 4-band features vs. the established 8-layer model")
    print("=" * 80)
    print(f"{'(reference) Full 8-layer model':<40s} acc={FULL_MODEL_LOOCV_ACCURACY:.3f}  "
          f"auc=n/a (see earlier Wilson-CI analysis)\n")

    lr_result = report("4-band, Logistic Regression", X, y, lambda: LogisticRegression(max_iter=1000))
    rf_result = report("4-band, Random Forest", X, y,
                        lambda: RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42))

    print("\nPer-band group means (central-window, microns):")
    for col, name in zip(feature_cols, BAND_NAMES):
        hc_mean = df[df.label == 0][col].mean()
        ms_mean = df[df.label == 1][col].mean()
        print(f"  {name:12s}  HC={hc_mean:6.1f}  MS={ms_mean:6.1f}  diff={ms_mean - hc_mean:+6.1f}")

    print("\nInterpretation:")
    print("  Collapsing 8 layers into 4 bands merges RNFL+GCIP+INL into one number and")
    print("  OPL+ONL+IS into another -- exactly the merge that destroys the RNFL/GCIP-")
    print("  specific thinning signal the full model relies on (MS thins RNFL/GCIP but")
    print("  not INL in the same way, so averaging them together dilutes the signal).")
    print("  If accuracy here is noticeably below 0.657, that dilution is the likely")
    print("  reason, not a bug -- it's the expected cost of matching OCT5k's coarser")
    print("  annotation scheme, and should be stated as a limitation alongside any")
    print("  specificity number this model later produces on OCT5k subjects.")

    # Fit the final model on ALL 35 subjects (no held-out fold) for downstream use
    # against OCT5k's external AMD/DME/healthy subjects.
    best_name, best_fn, best_acc = (
        "logreg", lambda: LogisticRegression(max_iter=1000), lr_result["accuracy"]
    ) if lr_result["accuracy"] >= rf_result["accuracy"] else (
        "random_forest", lambda: RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42), rf_result["accuracy"]
    )
    scaler = StandardScaler().fit(X)
    final_model = best_fn()
    final_model.fit(scaler.transform(X), y)

    joblib.dump({
        "model": final_model,
        "scaler": scaler,
        "model_type": best_name,
        "feature_cols": feature_cols,
        "band_names": BAND_NAMES,
        "loocv_accuracy": best_acc,
        "trained_on_n": len(df),
    }, out_model)
    print(f"\nSaved comparison model ({best_name}, LOOCV acc={best_acc:.3f}, "
          f"trained on all {len(df)} subjects) to: {out_model}")
    print("This is the file run_oct5k_specificity.py will load once the OCT5k")
    print("AMD/DME/healthy files are available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to oct5k_4band_features.csv")
    parser.add_argument("--out-model", default="4band_comparison_model.joblib")
    args = parser.parse_args()
    main(args.csv_path, args.out_model)
