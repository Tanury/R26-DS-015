"""
backend/services/ms_mri_inference.py

Inference service for Multiple Sclerosis -- MRI (MindGlide region-volume) branch.

Pipeline: uploaded raw MRI (.nii/.nii.gz, NO preprocessing required -- that's
MindGlide's whole design point) -> MindGlide segmentation -> all 19 region
volumes -> 5 literature-motivated regions normalized by total brain volume
(same feature set validated in mri/multiple_sclerosis/build_multifeature_
classifier.py) -> trained VisionEncoder (mri/multiple_sclerosis/models/
vision_encoder.py) -> MS/HC prediction + region-volume comparison against
the training cohort's HC/MS group means (embedded in the checkpoint).
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mindglide import segment, volumes_dataframe
from mri.multiple_sclerosis.models.vision_encoder import build_encoder
from backend.services.ms_visualization import render_input_vs_segmentation

MRI_CHECKPOINT = _PROJECT_ROOT / "mri" / "multiple_sclerosis" / "models" / "mri_ms_encoder.pt"

# Must match mri/multiple_sclerosis/train_encoder.py exactly
RAW_REGIONS_FOR_TOTAL = [
    "CSF", "Ventricles_3_4_5", "DGM", "Pons", "Brainstem", "Cerebellum",
    "Temporal_lobe", "Temporal_horn_lateral_ventricle", "Lateral_ventricle",
    "Optic_chiasm", "Cerebellar_vermis", "Corpus_callosum", "White_matter",
    "Frontal_lobe_GM", "Limbic_cortex_GM", "Parietal_lobe_GM",
    "Occipital_lobe_GM", "Lesion", "Ventral_diencephalon",
]
LITERATURE_SUBSET = ["Lesion", "Lateral_ventricle", "DGM", "Corpus_callosum", "White_matter"]

MRI_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MRI_SW_BATCH_SIZE = 1  # matches what proved stable during validation -- avoids MPS OOM

_model: Optional[nn.Module] = None
_checkpoint: Optional[dict] = None


def _is_mri_available() -> bool:
    return MRI_CHECKPOINT.exists()


def _load_checkpoint() -> None:
    global _model, _checkpoint
    if _model is not None:
        return
    if not _is_mri_available():
        raise FileNotFoundError(
            f"MRI-MS model checkpoint not found at {MRI_CHECKPOINT}. "
            "Run mri/multiple_sclerosis/train_encoder.py first."
        )
    ckpt = torch.load(MRI_CHECKPOINT, map_location="cpu", weights_only=False)
    # This encoder is tiny (~39K params) -- always run it on CPU regardless of
    # what device MindGlide's segmentation used, same reasoning as the OCT branch.
    model = build_encoder(num_classes=2, feature_dim=len(ckpt["feature_cols"]),
                           device=torch.device("cpu"), verbose=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _model = model
    _checkpoint = ckpt


def run_mri_inference(file_bytes: bytes, filename: str) -> dict:
    """Run the full MRI pipeline on an uploaded raw scan and return a dict
    matching MSAnalysisResponse."""
    _load_checkpoint()
    assert _model is not None and _checkpoint is not None  # _load_checkpoint() guarantees this

    with tempfile.TemporaryDirectory() as tmp:
        # Preserve the real extension (.nii vs .nii.gz) so MindGlide/nibabel parse it correctly
        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        input_path = Path(tmp) / f"upload{suffix}"
        input_path.write_bytes(file_bytes)

        seg_path = Path(tmp) / "seg.nii.gz"
        segment(str(input_path), str(seg_path), device=MRI_DEVICE, sw_batch_size=MRI_SW_BATCH_SIZE)

        vol_df = volumes_dataframe(str(seg_path))

        try:
            slice_image = render_input_vs_segmentation(str(input_path), str(seg_path))
        except Exception:
            # Visualization is a nice-to-have -- never let a rendering failure
            # take down the actual prediction, which is the important part.
            slice_image = None

    volumes_by_region = dict(zip(vol_df["Region_Name"], vol_df["Volume_mm3"]))
    total_brain_volume = sum(volumes_by_region.get(r, 0.0) for r in RAW_REGIONS_FOR_TOTAL)
    if total_brain_volume <= 0:
        raise ValueError("MindGlide returned zero total brain volume -- segmentation likely failed.")

    feature_values = [volumes_by_region.get(r, 0.0) / total_brain_volume for r in LITERATURE_SUBSET]

    import numpy as np
    raw = np.array(feature_values).reshape(1, -1)
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

    hc_means = _checkpoint.get("hc_mean_per_feature")
    ms_means = _checkpoint.get("ms_mean_per_feature")
    regions = []
    for i, name in enumerate(LITERATURE_SUBSET):
        entry = {"region": name, "value_frac": float(feature_values[i])}
        if hc_means is not None:
            entry["hc_mean"] = float(hc_means[i])
        if ms_means is not None:
            entry["ms_mean"] = float(ms_means[i])
        regions.append(entry)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "class_scores": scores,
        "modality": "MRI (MindGlide Region Volumes)",
        "regions": regions,
        "slice_image": slice_image,
    }