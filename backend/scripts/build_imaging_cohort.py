"""Build the neuroimaging cohort store the fusion engine reads.

Runs the trained vision encoders over the processed per-subject features that
the neuroimaging module already extracted, so subjects can be fused without
re-uploading raw scans:

    MS OCT  eye/ms/preprocessing/thickness_features.csv      -> eye/ms/models/oct_thickness_encoder.pt
    MS MRI  mri/multiple_sclerosis/mindglide_all_features.csv -> mri/multiple_sclerosis/models/mri_ms_encoder.pt

AD and PD have no processed features in the repository (the AD test split
only stores Colab paths), so they are not part of the cohort.

Both production checkpoints were fit on every subject in these CSVs, so the
predictions are in-sample. Each record carries the model's LOOCV metrics and
`in_sample: true` so the UI can say so.

    python backend/scripts/build_imaging_cohort.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eye.ms.models.vision_encoder import build_encoder as build_oct_encoder  # noqa: E402
from mri.multiple_sclerosis.models.vision_encoder import build_encoder as build_mri_encoder  # noqa: E402

OUTPUT = REPO_ROOT / "backend" / "app" / "models" / "imaging" / "cohort.json"
CPU = torch.device("cpu")


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _run(model: torch.nn.Module, ckpt: dict, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = (features - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
    with torch.no_grad():
        z_img, logits = model(torch.tensor(scaled, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    return z_img.numpy(), probs


def _records(prefix: str, scan_type: str, modality: str, ids, labels, z_img, probs, ckpt, feature_rows) -> list[dict]:
    records = []
    for subject, label, emb, (p_hc, p_ms), features in zip(ids, labels, z_img, probs, feature_rows):
        prediction = "MS" if p_ms >= p_hc else "HC"
        records.append({
            "subject_id": f"{prefix}-{subject}",
            "condition": "MS",
            "scan_type": scan_type,
            "modality": modality,
            "true_label": "MS" if int(label) == 1 else "HC",
            "prediction": prediction,
            "confidence": round(float(max(p_hc, p_ms)) * 100, 2),
            "risk_level": "HIGH" if prediction == "MS" else "LOW",
            "probabilities": {"MS": round(float(p_ms), 6), "HC": round(float(p_hc), 6)},
            "features": {name: round(float(value), 6) for name, value in features.items()},
            "embedding_dim": int(emb.shape[0]),
            "embedding": [round(float(v), 5) for v in emb],
            "in_sample": True,
            "model_metrics": {
                "loocv_accuracy": round(float(ckpt["loocv_accuracy"]), 4),
                "loocv_roc_auc": round(float(ckpt["loocv_roc_auc"]), 4),
            },
        })
    return records


def build_oct() -> list[dict]:
    ckpt = _load(REPO_ROOT / "eye" / "ms" / "models" / "oct_thickness_encoder.pt")
    model = build_oct_encoder(num_classes=2, thickness_dim=len(ckpt["feature_cols"]), device=CPU, verbose=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    df = pd.read_csv(REPO_ROOT / "eye" / "ms" / "preprocessing" / "thickness_features.csv")
    features = df[ckpt["feature_cols"]]
    z_img, probs = _run(model, ckpt, features.to_numpy(dtype=np.float64))
    rows = [dict(zip(ckpt["layer_names"], row)) for row in features.to_numpy()]
    return _records("MS-OCT", "Retinal OCT", "OCT (Retinal Thickness)", df["subject_id"], df["label"], z_img, probs, ckpt, rows)


def build_mri() -> list[dict]:
    ckpt = _load(REPO_ROOT / "mri" / "multiple_sclerosis" / "models" / "mri_ms_encoder.pt")
    model = build_mri_encoder(num_classes=2, feature_dim=len(ckpt["feature_cols"]), device=CPU, verbose=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    df = pd.read_csv(REPO_ROOT / "mri" / "multiple_sclerosis" / "mindglide_all_features.csv", dtype={"id": str})
    # Same normalisation as mri/multiple_sclerosis/train_encoder.py
    total = df[ckpt["raw_regions_for_total"]].sum(axis=1)
    for column in ckpt["feature_cols"]:
        df[column] = df[column.removesuffix("_frac")] / total
    features = df[ckpt["feature_cols"]]
    z_img, probs = _run(model, ckpt, features.to_numpy(dtype=np.float64))
    rows = [dict(zip(ckpt["feature_cols"], row)) for row in features.to_numpy()]
    return _records("MS-MRI", "Brain MRI (MindGlide)", "MRI (MindGlide Region Volumes)", df["id"], df["label"], z_img, probs, ckpt, rows)


def main() -> None:
    subjects = build_oct() + build_mri()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conditions_available": ["MS"],
        "conditions_missing": {
            "AD": "No processed AD features or trained checkpoint in the repository.",
            "PD": "No processed PD features in the repository.",
        },
        "subjects": subjects,
    }
    OUTPUT.write_text(json.dumps(payload), encoding="utf-8")
    correct = sum(s["prediction"] == s["true_label"] for s in subjects)
    print(f"Wrote {len(subjects)} subjects to {OUTPUT} ({correct}/{len(subjects)} in-sample correct)")


if __name__ == "__main__":
    main()
