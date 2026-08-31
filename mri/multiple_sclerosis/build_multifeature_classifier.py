"""
build_multifeature_classifier.py

Extends the single-feature (lesion volume alone) analysis to use
MindGlide's full set of region volumes. Runs a confound check FIRST,
since MS3SEG and LEMON are two independent studies (different country,
scanner, population) that could differ in overall brain/head size for
reasons unrelated to MS -- if so, a classifier could partly be learning
"which study is this" rather than "does this brain show MS pathology".

Feature sets compared (all via LOOCV, same methodology as the OCT branch
and the single-feature lesion-volume check):
  1. Lesion volume alone (the established baseline, AUC 0.974 on this data)
  2. Total brain volume alone (the confound check -- if this alone
     separates the groups well, that's a warning sign, not a good result)
  3. All 19 raw region volumes (kitchen-sink)
  4. All 19 region volumes normalized by total brain volume (removes the
     head-size/dataset-scale confound)
  5. A literature-motivated subset: Lesion + Lateral_ventricle + DGM +
     Corpus_callosum + White_matter (established MS atrophy/lesion
     markers, chosen a priori from the literature, not selected by
     looking at which features happen to separate these groups best)

Usage:
    python3 build_multifeature_classifier.py mindglide_all_features.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ALL_REGIONS = [
    "CSF", "Ventricles_3_4_5", "DGM", "Pons", "Brainstem", "Cerebellum",
    "Temporal_lobe", "Temporal_horn_lateral_ventricle", "Lateral_ventricle",
    "Optic_chiasm", "Cerebellar_vermis", "Corpus_callosum", "White_matter",
    "Frontal_lobe_GM", "Limbic_cortex_GM", "Parietal_lobe_GM",
    "Occipital_lobe_GM", "Lesion", "Ventral_diencephalon",
]
LITERATURE_SUBSET = ["Lesion", "Lateral_ventricle", "DGM", "Corpus_callosum", "White_matter"]


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


def report(name: str, df: pd.DataFrame, features: list) -> dict:
    X = df[features].values
    y = df["label"].values
    lr = run_loocv(X, y, lambda: LogisticRegression(max_iter=1000))
    rf = run_loocv(X, y, lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
    print(f"{name:<45s} LogReg: acc={lr['accuracy']:.3f} auc={lr['roc_auc']:.3f}   "
          f"RF: acc={rf['accuracy']:.3f} auc={rf['roc_auc']:.3f}")
    return {"lr": lr, "rf": rf}


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path, dtype={"id": str})
    print(f"Loaded {len(df)} subjects: {(df['label']==0).sum()} Healthy, {(df['label']==1).sum()} MS\n")

    # Total brain volume = sum of all non-background regions
    df["total_brain_volume"] = df[ALL_REGIONS].sum(axis=1)

    print("=" * 90)
    print("STEP 1 -- CONFOUND CHECK: does total brain volume ALONE separate the groups?")
    print("=" * 90)
    print("(If this AUC is also very high, some of the lesion-volume result may reflect")
    print(" dataset/study differences rather than pure disease signal.)\n")
    confound_result = report("Total brain volume alone", df, ["total_brain_volume"])
    print(f"\nMean total brain volume -- MS3SEG: {df[df.label==1]['total_brain_volume'].mean():.0f} mm³  |  "
          f"LEMON: {df[df.label==0]['total_brain_volume'].mean():.0f} mm³")

    print("\n" + "=" * 90)
    print("STEP 2 -- FEATURE SET COMPARISON (all via LOOCV)")
    print("=" * 90)
    report("Lesion volume alone (baseline)", df, ["Lesion"])
    report("All 19 raw region volumes", df, ALL_REGIONS)

    df_norm = df.copy()
    for r in ALL_REGIONS:
        df_norm[r + "_frac"] = df_norm[r] / df_norm["total_brain_volume"]
    norm_features = [r + "_frac" for r in ALL_REGIONS]
    report("All 19 regions, normalized by brain volume", df_norm, norm_features)

    report("Literature subset (Lesion+Ventricle+DGM+CC+WM)", df, LITERATURE_SUBSET)

    df_norm_lit = df.copy()
    for r in LITERATURE_SUBSET:
        df_norm_lit[r + "_frac"] = df_norm_lit[r] / df_norm_lit["total_brain_volume"]
    lit_norm_features = [r + "_frac" for r in LITERATURE_SUBSET]
    report("Literature subset, normalized by brain volume (FINAL CANDIDATE)", df_norm_lit, lit_norm_features)

    print("\nInterpretation:")
    print("  If 'Total brain volume alone' has a high AUC, treat the multi-feature results")
    print("  with caution -- they may partly reflect which dataset a scan came from, not MS")
    print("  pathology specifically. The normalized-features result is the more trustworthy")
    print("  multi-feature number in that case, since dividing by total brain volume removes")
    print("  most of that scale confound.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to mindglide_all_features.csv")
    args = parser.parse_args()
    main(args.csv_path)