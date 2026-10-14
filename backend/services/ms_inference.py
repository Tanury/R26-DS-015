"""
backend/services/ms_inference.py

Inference service for Multiple Sclerosis -- OCT (retinal thickness) branch.

Pipeline: uploaded .vol + .mat -> vol_loader/mat_loader -> central-window
thickness features (same 10-B-scan foveal window used in training) ->
trained ThicknessEncoder (VisionEncoder, eye/ms/models/vision_encoder.py)
-> MS/HC prediction + per-layer thickness compared against the training
cohort's HC/MS group means (embedded in the checkpoint at training time).

The MRI branch for MS is intentionally NOT implemented here yet --
deferred, tracked separately. Only the OCT modality is served.
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Make the eye/ package importable regardless of the backend's run directory
# (mirrors how mri/alzheimers and mri/parkinsons are presumably imported --
# adjust this path if your project layout differs).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eye.ms.preprocessing.vol_loader import load_vol
from eye.ms.preprocessing.mat_loader import load_boundaries
from eye.ms.preprocessing.thickness import compute_thickness_features, central_bscan_window
from eye.ms.models.vision_encoder import build_encoder
#new
from backend.services.oct_visualization import render_bscan_with_boundaries

OCT_CHECKPOINT = _PROJECT_ROOT / "eye" / "ms" / "models" / "oct_thickness_encoder.pt"
CENTRAL_WINDOW_SIZE = 10  # must match what train_thickness_encoder.py was trained on

_model: Optional[nn.Module] = None
_checkpoint: Optional[dict] = None


def _is_oct_available() -> bool:
    return OCT_CHECKPOINT.exists()


def _is_mri_available() -> bool:
    return False  # MS MRI branch deferred


def _load_checkpoint() -> None:
    global _model, _checkpoint
    if _model is not None:
        return
    if not _is_oct_available():
        raise FileNotFoundError(
            f"OCT model checkpoint not found at {OCT_CHECKPOINT}. "
            "Run eye/ms/train_thickness_encoder.py first."
        )
    ckpt = torch.load(OCT_CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_encoder(
        num_classes=2, thickness_dim=len(ckpt["feature_cols"]), device=torch.device("cpu"), verbose=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _model = model
    _checkpoint = ckpt


def run_oct_inference(
    vol_bytes: bytes, vol_filename: str, mat_bytes: bytes, mat_filename: str
) -> dict:
    """Run the full OCT pipeline on uploaded .vol + .mat file bytes and
    return a dict matching MSAnalysisResponse."""
    _load_checkpoint()
    assert _model is not None and _checkpoint is not None  # _load_checkpoint() guarantees this

    with tempfile.TemporaryDirectory() as tmp:
        vol_path = Path(tmp) / (vol_filename or "upload.vol")
        mat_path = Path(tmp) / (mat_filename or "upload.mat")
        vol_path.write_bytes(vol_bytes)
        mat_path.write_bytes(mat_bytes)

        vol = load_vol(str(vol_path))
        bd = load_boundaries(str(mat_path), size_x=vol.meta["size_x"])
        window = central_bscan_window(vol.meta["n_bscans"], window=CENTRAL_WINDOW_SIZE)
        feats = compute_thickness_features(
            bd.boundaries, scale_y_mm=vol.meta["scale_y"], bscan_indices=window
        )

        #new - try 
        try:
            import numpy as np
            mid_frame = window[len(window) // 2]
            bscan_clean = np.where(np.isnan(vol.volume[mid_frame]), 0.0, vol.volume[mid_frame])
            slice_image = render_bscan_with_boundaries(
                bscan_clean, bd.boundaries[mid_frame], feats.layer_names
                )
        except Exception:
            slice_image = None  #incase of a rendering issue, prediction wil still work

        #end of new part

    raw = feats.subject_vector.reshape(1, -1)
    scaled = (raw - _checkpoint["scaler_mean"]) / _checkpoint["scaler_scale"]

    with torch.no_grad():
        x = torch.tensor(scaled, dtype=torch.float32)
        _, logits = _model(x)
        proba = torch.softmax(logits, dim=1)[0]  # [p_healthy, p_ms]

    p_hc, p_ms = float(proba[0]), float(proba[1])
    prediction = "MS" if p_ms >= p_hc else "HC"
    confidence = max(p_hc, p_ms) * 100
    risk_level = "HIGH" if prediction == "MS" else "LOW"

    scores = [
        {"label": "Multiple Sclerosis", "confidence": p_ms},
        {"label": "Healthy Control", "confidence": p_hc},
    ]
    scores.sort(key=lambda s: s["confidence"], reverse=True)

    hc_means = _checkpoint.get("hc_mean_per_layer")
    ms_means = _checkpoint.get("ms_mean_per_layer")
    thickness = []
    for i, name in enumerate(feats.layer_names):
        entry = {"layer": name, "value_um": float(feats.subject_vector[i])}
        if hc_means is not None:
            entry["hc_mean"] = float(hc_means[i])
        if ms_means is not None:
            entry["ms_mean"] = float(ms_means[i])
        thickness.append(entry)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "class_scores": scores,
        "modality": "OCT (Retinal Thickness)",
        "thickness": thickness,
        #"slice_image": slice_image,  # new - remove if not working!!
    }