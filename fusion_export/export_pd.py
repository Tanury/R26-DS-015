"""
export_pd.py
============
PD DaTscan branch -> decision-level fusion handoff.

Source: mri/parkinsons/embeddings/dat_test_embeddings.npz, written by the
Colab DaTscan notebook (DaTEncoder, dat_best_model.pth, epoch 17) on the
held-out PPMI test set (282 subjects, one scan each).

There is no cross-validation for this branch, so Platt calibration is fitted
leave-one-out WITHIN the test set: each subject's calibrated probability comes
from a mapping fitted on the other 281. The mapping fitted on all 282 is saved
for use on new patients.

Outputs (outputs/fusion_export/pd_datscan/):
  predictions.csv   subject_id, label, P_PD_raw, P_PD_calibrated, pred
  z_img.npz         embeddings (282,256) + subject_ids, labels, probs
  metrics.json      accuracy / ROC-AUC / sensitivity / specificity / Brier,
                    raw and calibrated, + Platt params

Run from final_neuro/:
    python fusion_export/export_pd.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mri/parkinsons/embeddings/dat_test_embeddings.npz"
OUT = ROOT / "outputs/fusion_export/pd_datscan"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "sensitivity_pd": float(((pred == 1) & (y == 1)).sum() / (y == 1).sum()),
        "specificity_hc": float(((pred == 0) & (y == 0)).sum() / (y == 0).sum()),
        "brier": float(brier_score_loss(y, p)),
    }


def main():
    d = np.load(SRC, allow_pickle=True)
    ids = d["subject_ids"].astype(str)
    y = (d["labels"] == "PD").astype(int)
    raw = d["pd_probs"].astype(float)
    emb = d["embeddings"].astype(np.float32)
    assert len(set(ids)) == len(ids), "duplicate subject IDs in test set"

    z, cal = logit(raw), np.zeros(len(y))
    for tr, te in LeaveOneOut().split(z):
        cal[te] = LogisticRegression(C=1e6).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
    full = LogisticRegression(C=1e6).fit(z, y)
    platt = {"a": float(full.coef_[0, 0]), "b": float(full.intercept_[0]),
             "apply_as": "P_cal = sigmoid(a * logit(P_raw) + b)"}

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_id": ids, "label": y, "label_name": np.where(y == 1, "PD", "HC"),
                  "P_PD_raw": raw.round(6), "P_PD_calibrated": cal.round(6),
                  "pred": (cal >= 0.5).astype(int)}).to_csv(OUT / "predictions.csv", index=False)
    np.savez(OUT / "z_img.npz", embeddings=emb, subject_ids=ids, labels=y,
             probs_raw=raw.astype(np.float32), class_names=np.array(["HC", "PD"]))

    r, c = scores(y, raw), scores(y, cal)
    metrics = {
        "branch": "pd_datscan",
        "modality": "DaTscan SPECT (2D striatal slice)",
        "dataset": "PPMI",
        "model": "DaTEncoder 2D CNN -> 256-d L2-normalised z_img, linear head",
        "checkpoint": "mri/parkinsons/models/checkpoints/dat_best_model.pth (epoch 17)",
        "evaluation": "single held-out test set, one scan per subject; "
                      "calibration fitted leave-one-out within the test set",
        "n": int(len(y)), "n_pd": int(y.sum()), "n_hc": int((1 - y).sum()),
        "majority_class_rate": float(max(y.mean(), 1 - y.mean())),
        "test_raw": r, "test_calibrated": c, "platt": platt,
        "note": "Train/test subject overlap could not be re-checked here (train split not "
                "saved locally); the notebook split by subject.",
        "embedding_norm_mean": float(np.linalg.norm(emb, axis=1).mean()),
        "exported": str(date.today()),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for name, s in [("raw", r), ("calibrated", c)]:
        print(f"{name:<11} acc {s['accuracy']:.3f}  AUC {s['roc_auc']:.3f}  "
              f"sens {s['sensitivity_pd']:.3f}  spec {s['specificity_hc']:.3f}  Brier {s['brier']:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
