"""
oct_thickness_classifier.py
=============================
R26-DS-015 — eye/ms

MS vs. Healthy classifier on the 8-d retinal-layer thickness features
(from thickness_features.csv, produced by batch_preprocess_thickness.py).

With n=35 subjects, Leave-One-Out Cross-Validation (LOOCV) is used rather
than a fixed train/test split -- holding out a fixed test set from 35
subjects wastes too much data and gives an unstable estimate. LOOCV is
also standard practice for cohorts this size in the OCT/MS literature
(the reference dataset's own validation work used the same approach).

Two models are compared:
  - Logistic Regression (interpretable coefficients -- which layers
    actually drive the prediction)
  - Random Forest (captures non-linear layer interactions, at the cost
    of interpretability)

Usage:
    python3 oct_thickness_classifier.py thickness_features.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_WHOLE = [
    "thickness_RNFL_um", "thickness_GCIP_um", "thickness_INL_um",
    "thickness_OPL_um", "thickness_ONL_um", "thickness_IS_um",
    "thickness_OS_um", "thickness_RPE_um",
]
FEATURE_COLS_CENTRAL = [c.replace("_um", "_central_um") for c in FEATURE_COLS_WHOLE]

# Literature-grounded ratio features (computed from central-window thickness,
# not re-derived from raw scans). Inner-vs-outer retina distinction follows
# the established grouping in the OCT/MS literature (e.g. Saidha et al.) --
# MS causes disproportionate inner-retinal thinning while the outer retina
# (mostly photoreceptors) is comparatively spared.
RATIO_COLS = ["ratio_GCIP_RNFL", "ratio_GCIP_INL", "ratio_inner_outer"]


def add_ratio_features(df: pd.DataFrame, suffix: str = "_central_um") -> pd.DataFrame:
    df = df.copy()
    rnfl = df[f"thickness_RNFL{suffix}"]
    gcip = df[f"thickness_GCIP{suffix}"]
    inl = df[f"thickness_INL{suffix}"]
    opl = df[f"thickness_OPL{suffix}"]
    onl = df[f"thickness_ONL{suffix}"]
    is_ = df[f"thickness_IS{suffix}"]
    os_ = df[f"thickness_OS{suffix}"]
    rpe = df[f"thickness_RPE{suffix}"]

    inner_retina = rnfl + gcip + inl + opl   # RNFL+GCIP+INL+OPL -- predominantly affected in MS
    outer_retina = onl + is_ + os_ + rpe     # ONL+IS+OS+RPE -- photoreceptor complex, relatively spared

    df["ratio_GCIP_RNFL"] = gcip / rnfl
    df["ratio_GCIP_INL"] = gcip / inl
    df["ratio_inner_outer"] = inner_retina / outer_retina
    return df


def run_loocv(X: np.ndarray, y: np.ndarray, model_fn) -> dict:
    """Leave-One-Out CV: fit on 34, predict the 1 held out, repeat for all 35."""
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler on training fold only (avoids leaking test-fold stats)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]
        proba = model.predict_proba(X_test_s)[0, 1]

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_proba.append(proba)

    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def fit_full_logreg_for_coefficients(X: np.ndarray, y: np.ndarray) -> tuple:
    """Fit on ALL 35 subjects (not LOOCV) just to inspect which layers
    drive the decision -- this is for interpretation only, never report
    this model's own training accuracy as a performance metric."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_s, y)
    return model.coef_[0], model


def evaluate_feature_set(df: pd.DataFrame, feature_cols: list, label: str) -> dict:
    X = df[feature_cols].values
    y = df["label"].values
    layer_names = [c.replace("thickness_", "").replace("_central_um", "").replace("_um", "")
                   for c in feature_cols]

    print("=" * 70)
    print(f"{label} -- LOGISTIC REGRESSION (LOOCV)")
    print("=" * 70)
    lr_results = run_loocv(X, y, lambda: LogisticRegression(max_iter=1000))
    print(f"Accuracy: {lr_results['accuracy']:.3f}  |  ROC-AUC: {lr_results['roc_auc']:.3f}")
    print(f"Confusion matrix (rows=true, cols=pred, 0=Healthy 1=MS):")
    print(lr_results["confusion_matrix"])

    print(f"\n{label} -- RANDOM FOREST (LOOCV)")
    rf_results = run_loocv(
        X, y, lambda: RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
    )
    print(f"Accuracy: {rf_results['accuracy']:.3f}  |  ROC-AUC: {rf_results['roc_auc']:.3f}")
    print(f"Confusion matrix (rows=true, cols=pred, 0=Healthy 1=MS):")
    print(rf_results["confusion_matrix"])

    print(f"\n{label} -- FEATURE IMPORTANCE (logistic regression coefficients, standardized)")
    coefs, _ = fit_full_logreg_for_coefficients(X, y)
    for name, coef in sorted(zip(layer_names, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:6s}: {coef:+.3f}")

    misclassified = df.loc[lr_results["y_true"] != lr_results["y_pred"], "subject_id"].tolist()
    print(f"\n{label} -- Misclassified (Logistic Regression, LOOCV): {misclassified or 'None'}")
    print()

    return {"lr": lr_results, "rf": rf_results}


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} subjects: {(df['label'] == 0).sum()} Healthy, "
          f"{(df['label'] == 1).sum()} MS\n")

    has_central = all(c in df.columns for c in FEATURE_COLS_CENTRAL)

    whole_results = evaluate_feature_set(df, FEATURE_COLS_WHOLE, "WHOLE-SCAN (all 49 B-scans)")

    if has_central:
        central_results = evaluate_feature_set(
            df, FEATURE_COLS_CENTRAL, "CENTRAL-WINDOW (foveal B-scans)"
        )

        df_ratios = add_ratio_features(df, suffix="_central_um")
        ratios_only_results = evaluate_feature_set(
            df_ratios, RATIO_COLS, "CENTRAL-WINDOW RATIOS ONLY (GCIP/RNFL, GCIP/INL, inner/outer)"
        )
        central_plus_ratios_results = evaluate_feature_set(
            df_ratios, FEATURE_COLS_CENTRAL + RATIO_COLS,
            "CENTRAL-WINDOW + RATIOS (11 features)"
        )

        print("=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        rows = [
            ("Whole-scan", whole_results),
            ("Central-window", central_results),
            ("Central ratios only", ratios_only_results),
            ("Central + ratios", central_plus_ratios_results),
        ]
        print(f"{'Feature set':<25s}{'LogReg Acc':<14s}{'LogReg AUC':<14s}{'RF Acc':<12s}{'RF AUC':<10s}")
        for name, res in rows:
            print(f"{name:<25s}{res['lr']['accuracy']:<14.3f}{res['lr']['roc_auc']:<14.3f}"
                  f"{res['rf']['accuracy']:<12.3f}{res['rf']['roc_auc']:<10.3f}")
    else:
        print("(No '_central_um' columns found in CSV -- only whole-scan results shown. "
              "Re-run batch_preprocess_thickness.py with the updated version to get both.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to thickness_features.csv")
    args = parser.parse_args()
    main(args.csv_path)