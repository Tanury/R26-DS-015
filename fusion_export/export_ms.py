"""
export_ms.py
============
MS branches (brain MRI + retinal OCT) -> decision-level fusion handoff.

For each branch it:
  1. Re-runs LOOCV with the SAME model, features and training settings as the
     branch's training script, and keeps every held-out probability
     (out-of-fold, OOF). These are the numbers the fusion engine should learn
     its weights from.
  2. Calibrates the OOF probabilities with Platt scaling, itself evaluated by
     LOOCV so the calibrated numbers stay honest. The Platt parameters fitted
     on all OOF probabilities are saved so the same mapping can be applied to
     new patients at inference time.
  3. Loads the production checkpoint (models/*.pt) and writes the 256-d z_img
     for every subject, plus per-subject feature attributions
     (gradient x input on P(MS)).

Outputs (outputs/fusion_export/<branch>/):
  predictions_oof.csv   subject_id, label, P_MS_raw_oof, P_MS_calibrated_oof, pred
  features.csv          the exact model input features (unscaled) per subject
  attributions.csv      per-subject contribution of each feature to P(MS)
  z_img.npz             embeddings (N,256) from the production checkpoint
  metrics.json          LOOCV accuracy / ROC-AUC / sensitivity / specificity /
                        Brier, before and after calibration, + Platt params

Run from final_neuro/ with the project venv active:
    python fusion_export/export_ms.py            # both branches
    python fusion_export/export_ms.py --branch ms_oct
"""

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "fusion_export"
SEED = 42


def load_module(path: Path, name: str):
    """Import a training script by path (its folder must be importable for models/)."""
    # Both branches have a folder called models/, so clear any earlier one
    # from the import cache AND from sys.path before loading this branch.
    def _clear():
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        for d in [str(b["train_script"].parent) for b in BRANCHES.values()]:
            while d in sys.path:
                sys.path.remove(d)
    _clear()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _clear()
    return mod


BRANCHES = {
    "ms_mri": {
        "train_script": ROOT / "mri/multiple_sclerosis/train_encoder.py",
        "csv": ROOT / "mri/multiple_sclerosis/mindglide_all_features.csv",
        "checkpoint": ROOT / "mri/multiple_sclerosis/models/mri_ms_encoder.pt",
        "id_col": "id",
        "modality": "brain MRI (MindGlide region volumes / total brain volume)",
        "dataset": "MS3SEG (MS) + LEMON (healthy controls)",
    },
    "ms_oct": {
        "train_script": ROOT / "eye/ms/train_thickness_encoder.py",
        "csv": ROOT / "eye/ms/preprocessing/thickness_features.csv",
        "checkpoint": ROOT / "eye/ms/models/oct_thickness_encoder.pt",
        "id_col": "subject_id",
        "modality": "retinal OCT (central-window layer thickness, 8 layers)",
        "dataset": "JHU / IACL HC-MS OCT (Spectralis macula)",
    },
}


def get_features(branch, mod, df):
    if branch == "ms_mri":
        df = mod.prepare_features(df)
        cols = list(mod.FEATURE_COLS)
    else:
        cols = list(mod.FEATURE_COLS_CENTRAL)
    return df, cols


def build_eval_model(branch, mod, n_features):
    if branch == "ms_mri":
        return mod.build_encoder(num_classes=2, feature_dim=n_features,
                                 device=torch.device("cpu"), verbose=False)
    return mod.build_encoder(num_classes=2, thickness_dim=n_features,
                             device=torch.device("cpu"), verbose=False)


def p_ms(model, X):
    with torch.no_grad():
        _, logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1)[:, 1].numpy()


def loocv_oof(mod, X, y, epochs):
    oof = np.zeros(len(y))
    for i, (tr, te) in enumerate(LeaveOneOut().split(X)):
        torch.manual_seed(SEED + i)
        sc = StandardScaler().fit(X[tr])
        m = mod.train_one_fold(sc.transform(X[tr]), y[tr], epochs=epochs)
        m.eval()
        oof[te] = p_ms(m, sc.transform(X[te]))
        print(f"  fold {i + 1}/{len(y)}", end="\r")
    print()
    return oof


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt_loocv(p, y):
    """Calibrated OOF probs: each subject's mapping is fitted without that subject."""
    z, cal = logit(p), np.zeros(len(y))
    for tr, te in LeaveOneOut().split(z):
        lr = LogisticRegression(C=1e6).fit(z[tr], y[tr])
        cal[te] = lr.predict_proba(z[te])[:, 1]
    full = LogisticRegression(C=1e6).fit(z, y)
    return cal, {"a": float(full.coef_[0, 0]), "b": float(full.intercept_[0]),
                 "apply_as": "P_cal = sigmoid(a * logit(P_raw) + b)"}


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "sensitivity_ms": tp / int((y == 1).sum()),
        "specificity_hc": tn / int((y == 0).sum()),
        "brier": float(brier_score_loss(y, p)),
    }


def attributions(model, Xs):
    """Gradient x input on P(MS), in standardised feature units."""
    x = torch.tensor(Xs, dtype=torch.float32, requires_grad=True)
    _, logits = model(x)
    torch.softmax(logits, dim=1)[:, 1].sum().backward()
    return (x.grad * x).detach().numpy()


def run(branch, epochs):
    cfg = BRANCHES[branch]
    print(f"\n=== {branch} ===")
    mod = load_module(cfg["train_script"], f"train_{branch}")
    df = pd.read_csv(cfg["csv"], dtype={cfg["id_col"]: str})
    df, cols = get_features(branch, mod, df)
    X, y = df[cols].values.astype(float), df["label"].values.astype(int)
    ids = df[cfg["id_col"]].astype(str).values
    print(f"{len(y)} subjects ({(y == 1).sum()} MS, {(y == 0).sum()} HC), {len(cols)} features")

    # 1-2. LOOCV OOF + calibration
    oof = loocv_oof(mod, X, y, epochs)
    cal, platt = platt_loocv(oof, y)

    # 3. production checkpoint -> embeddings + attributions
    ck = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = build_eval_model(branch, mod, len(cols))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    Xs = (X - np.asarray(ck["scaler_mean"])) / np.asarray(ck["scaler_scale"])
    with torch.no_grad():
        z, _ = model(torch.tensor(Xs, dtype=torch.float32))
    z = z.numpy().astype(np.float32)
    att = attributions(model, Xs)

    out = OUT_ROOT / branch
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_id": ids, "label": y,
                  "label_name": np.where(y == 1, "MS", "HC"),
                  "P_MS_raw_oof": oof.round(6), "P_MS_calibrated_oof": cal.round(6),
                  "pred": (cal >= 0.5).astype(int)}).to_csv(out / "predictions_oof.csv", index=False)
    pd.DataFrame(X, columns=cols).assign(subject_id=ids, label=y)[["subject_id", "label", *cols]] \
        .to_csv(out / "features.csv", index=False)
    pd.DataFrame(att.round(6), columns=cols).assign(subject_id=ids)[["subject_id", *cols]] \
        .to_csv(out / "attributions.csv", index=False)
    np.savez(out / "z_img.npz", embeddings=z, subject_ids=ids, labels=y,
             class_names=np.array(["HC", "MS"]))

    raw, calm = scores(y, oof), scores(y, cal)
    metrics = {
        "branch": branch, "modality": cfg["modality"], "dataset": cfg["dataset"],
        "n": int(len(y)), "n_ms": int((y == 1).sum()), "n_hc": int((y == 0).sum()),
        "majority_class_rate": float(max(y.mean(), 1 - y.mean())),
        "evaluation": "leave-one-out CV, scaler fitted inside each fold, seed 42",
        "features": cols,
        "loocv_raw": raw, "loocv_calibrated": calm, "platt": platt,
        "checkpoint": str(cfg["checkpoint"].relative_to(ROOT)),
        "checkpoint_saved_loocv": {"accuracy": float(ck.get("loocv_accuracy", float("nan"))),
                                   "roc_auc": float(ck.get("loocv_roc_auc", float("nan")))},
        "global_feature_importance_mean_abs_attr": dict(zip(cols, np.abs(att).mean(0).round(5).tolist())),
        "z_img_note": "from the production model fitted on ALL subjects, so in-sample for these subjects",
        "embedding_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
        "exported": str(date.today()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"LOOCV raw        acc {raw['accuracy']:.3f}  AUC {raw['roc_auc']:.3f}  "
          f"sens {raw['sensitivity_ms']:.3f}  spec {raw['specificity_hc']:.3f}  Brier {raw['brier']:.3f}")
    print(f"LOOCV calibrated acc {calm['accuracy']:.3f}  AUC {calm['roc_auc']:.3f}  "
          f"sens {calm['sensitivity_ms']:.3f}  spec {calm['specificity_hc']:.3f}  Brier {calm['brier']:.3f}")
    print(f"(checkpoint's own saved LOOCV: {metrics['checkpoint_saved_loocv']})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=[*BRANCHES, "all"], default="all")
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    for b in (BRANCHES if a.branch == "all" else [a.branch]):
        run(b, a.epochs)
