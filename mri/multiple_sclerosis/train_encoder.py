"""
train_encoder.py


Trains the MindGlide-region-based VisionEncoder (models/vision_encoder.py)
on mindglide_all_features.csv.

Two passes, same pattern as eye/ms/train_thickness_encoder.py:
  1. LOOCV -- honest performance estimate (comparable to the earlier
     LogReg/RF LOOCV results on the same 5 normalized features).
  2. Final fit on ALL subjects -- production model whose weights get
     saved and loaded by the fusion engine to produce z_img for new
     subjects. Its own training-set accuracy is NOT a valid performance
     metric -- only the LOOCV number from pass 1 should be reported.

Usage:
    python3 train_encoder.py mindglide_all_features.csv \\
        --out models/mri_ms_encoder.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from models.vision_encoder import VisionEncoder, build_encoder

RAW_REGIONS_FOR_TOTAL = [
    "CSF", "Ventricles_3_4_5", "DGM", "Pons", "Brainstem", "Cerebellum",
    "Temporal_lobe", "Temporal_horn_lateral_ventricle", "Lateral_ventricle",
    "Optic_chiasm", "Cerebellar_vermis", "Corpus_callosum", "White_matter",
    "Frontal_lobe_GM", "Limbic_cortex_GM", "Parietal_lobe_GM",
    "Occipital_lobe_GM", "Lesion", "Ventral_diencephalon",
]
LITERATURE_SUBSET = ["Lesion", "Lateral_ventricle", "DGM", "Corpus_callosum", "White_matter"]
FEATURE_COLS = [r + "_frac" for r in LITERATURE_SUBSET]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 normalized literature-subset features from raw region volumes."""
    df = df.copy()
    df["total_brain_volume"] = df[RAW_REGIONS_FOR_TOTAL].sum(axis=1)
    for r in LITERATURE_SUBSET:
        df[r + "_frac"] = df[r] / df["total_brain_volume"]
    return df


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int = 300, lr: float = 1e-3, verbose: bool = False
) -> VisionEncoder:
    device = torch.device("cpu")  # tiny model/dataset -- CPU is plenty
    model = build_encoder(num_classes=2, feature_dim=X_train.shape[1], device=device, verbose=verbose)
    model.train()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        _, logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    return model


def run_loocv(X: np.ndarray, y: np.ndarray, epochs: int = 300) -> dict:
    loo = LeaveOneOut()
    y_true, y_pred, y_proba = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = train_one_fold(X_train_s, y_train, epochs=epochs)
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
            _, logits = model(X_test_t)
            proba = torch.softmax(logits, dim=1)[0, 1].item()
            pred = int(proba >= 0.5)

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_proba.append(proba)

    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "y_true": y_true, "y_pred": y_pred,
    }


def main(csv_path: str, out_path: str, epochs: int = 300) -> None:
    raw_df = pd.read_csv(csv_path, dtype={"id": str})
    df = prepare_features(raw_df)
    X = df[FEATURE_COLS].values
    y = df["label"].values

    print(f"Loaded {len(df)} subjects: {(y == 0).sum()} Healthy, {(y == 1).sum()} MS\n")
    print(f"Features: {FEATURE_COLS}\n")

    print("=" * 70)
    print("PASS 1: LOOCV (honest performance estimate -- cite this number)")
    print("=" * 70)
    results = run_loocv(X, y, epochs=epochs)
    print(f"Accuracy: {results['accuracy']:.3f}  |  ROC-AUC: {results['roc_auc']:.3f}")
    print("(Compare against the earlier LogReg/RF LOOCV results on the same 5 features -- "
          "this MLP exists to produce the fusion-compatible z_img embedding, not to be a "
          "stronger standalone classifier than those.)")

    print("\n" + "=" * 70)
    print("PASS 2: Final fit on all subjects (production model for fusion engine)")
    print("=" * 70)
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X)
    final_model = train_one_fold(X_all_s, y, epochs=epochs, verbose=True)
    final_model.eval()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_cols": FEATURE_COLS,
            "raw_regions_for_total": RAW_REGIONS_FOR_TOTAL,
            "hc_mean_per_feature": X[y == 0].mean(axis=0),
            "ms_mean_per_feature": X[y == 1].mean(axis=0),
            "loocv_accuracy": results["accuracy"],
            "loocv_roc_auc": results["roc_auc"],
        },
        out_file,
    )
    print(f"Saved production model + scaler to: {out_file}")
    print(f"(Embedded LOOCV metrics for reference: "
          f"acc={results['accuracy']:.3f}, auc={results['roc_auc']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to mindglide_all_features.csv")
    parser.add_argument("--out", default="models/mri_ms_encoder.pt")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    main(args.csv_path, args.out, epochs=args.epochs)