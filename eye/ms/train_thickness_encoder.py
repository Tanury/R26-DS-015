"""
train_thickness_encoder.py
============================
R26-DS-015 — eye/ms

Trains the thickness-only VisionEncoder (models/vision_encoder.py) on
the central-window thickness features (thickness_features.csv).

Two passes:
  1. LOOCV -- honest performance estimate (comparable to the earlier
     Logistic Regression / Random Forest LOOCV results). Reports
     accuracy and ROC-AUC. This number is what you cite in the writeup.
  2. Final fit on ALL 35 subjects -- this is the production model whose
     weights get saved and loaded by the fusion engine to produce
     z_img_oct for new subjects. Its own training-set accuracy is NOT
     a valid performance metric (it has seen all the data) -- only the
     LOOCV number from pass 1 should be reported as this model's
     expected accuracy.

Usage:
    python3 train_thickness_encoder.py preprocessing/thickness_features.csv \\
        --out models/oct_thickness_encoder.pt
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

FEATURE_COLS_CENTRAL = [
    "thickness_RNFL_central_um", "thickness_GCIP_central_um", "thickness_INL_central_um",
    "thickness_OPL_central_um", "thickness_ONL_central_um", "thickness_IS_central_um",
    "thickness_OS_central_um", "thickness_RPE_central_um",
]


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int = 300, lr: float = 1e-3, verbose: bool = False
) -> VisionEncoder:
    device = torch.device("cpu")  # tiny model/dataset -- CPU is plenty
    model = build_encoder(num_classes=2, thickness_dim=X_train.shape[1], device=device, verbose=verbose)
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
            # BatchNorm needs >1 sample in train mode, but a single test
            # sample is fine in eval mode (uses running stats)
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
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main(csv_path: str, out_path: str, epochs: int = 300) -> None:
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS_CENTRAL].values
    y = df["label"].values

    print(f"Loaded {len(df)} subjects: {(y == 0).sum()} Healthy, {(y == 1).sum()} MS\n")

    print("=" * 70)
    print("PASS 1: LOOCV (honest performance estimate -- cite this number)")
    print("=" * 70)
    results = run_loocv(X, y, epochs=epochs)
    print(f"Accuracy: {results['accuracy']:.3f}  |  ROC-AUC: {results['roc_auc']:.3f}")
    print("(Compare against the earlier Logistic Regression / Random Forest LOOCV "
          "results on the same features -- this MLP is not expected to dramatically "
          "outperform those given n=35; it exists to produce the fusion-compatible "
          "z_img embedding, not to be a stronger standalone classifier.)")

    print("\n" + "=" * 70)
    print("PASS 2: Final fit on all 35 subjects (production model for fusion engine)")
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
            "feature_cols": FEATURE_COLS_CENTRAL,
            "layer_names": [c.replace("thickness_", "").replace("_central_um", "") for c in FEATURE_COLS_CENTRAL],
            "hc_mean_per_layer": X[y == 0].mean(axis=0),
            "ms_mean_per_layer": X[y == 1].mean(axis=0),
            "loocv_accuracy": results["accuracy"],
            "loocv_roc_auc": results["roc_auc"],
        },
        out_file,
    )
    print(f"Saved production model + scaler to: {out_file}")
    print(f"(Embedded LOOCV metrics for reference: "
          f"acc={results['accuracy']:.3f}, auc={results['roc_auc']:.3f})")
    print(f"(Embedded HC/MS group means per layer, for dashboard comparison bars)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to thickness_features.csv")
    parser.add_argument("--out", default="models/oct_thickness_encoder.pt")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    main(args.csv_path, args.out, epochs=args.epochs)