"""
backend/services/inference.py

Model loading and inference service for the Vision Encoder API.

Supports modes:
    1. MOCK mode  — returns realistic fake predictions (no model needed)
                    Used during development before model is trained.
    2. REAL mode  — loads best_model.pth and runs actual inference
                    Activated automatically when checkpoint exists.

Switch between modes by setting USE_MOCK = True/False below,
or by placing best_model.pth in the configured checkpoint path.
"""

import time
import sys
import tempfile
from nibabel import data
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Set to True to always use mock mode (for demo without trained model)
# Set to False to use real model when checkpoint is available
USE_MOCK = False

CHECKPOINT_PATH = "mri/alzheimers/models/checkpoints/best_model.pth"
IMAGE_SIZE      = (96, 96, 96)
CLASS_LABELS    = ["AD", "MCI", "CN"]
NUM_CLASSES     = 3

# Add project root to path for imports
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _is_model_available() -> bool:
    """Check if trained checkpoint exists."""
    return Path(CHECKPOINT_PATH).exists() and not USE_MOCK


def _load_model():
    """
    Load the VisionEncoder from checkpoint.
    Returns (model, device) tuple.
    Only called when real model is available.
    """
    import torch
    from mri.alzheimers.models.vision_encoder import build_encoder

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_encoder(num_classes=NUM_CLASSES, device=device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"  ✅  VisionEncoder loaded from {CHECKPOINT_PATH} on {device}")
    return model, device


def _preprocess_nifti(file_bytes: bytes):
    """
    Preprocess a NIfTI file for inference.
    Handles 3D and 4D volumes, various orientations.
    """
    import torch
    import nibabel as nib
    import numpy as np
    import tempfile
    from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        # Load and canonicalise orientation
        img  = nib.load(tmp_path)
        img  = nib.as_closest_canonical(img)
        data = np.asanyarray(img.dataobj, dtype=np.float32)

        # Handle 4D → take first volume
        if data.ndim == 4:
            data = data[..., 0]

        # Must be 3D at this point
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape}")

        # Manual scale to [0,1] before MONAI transforms
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        
        # Add channel dim manually — avoids EnsureChannelFirst issues
        # with non-standard NIfTI headers from ANTs output
        data = data[np.newaxis, ...]   # (1, H, W, D)

        # Now apply only Resize — ScaleIntensity already done above
        from monai.transforms import Resize
        resize = Resize(IMAGE_SIZE)
        tensor = resize(data)          # (1, 96, 96, 96)

        tensor = torch.as_tensor(
            np.array(tensor), dtype=torch.float32
        ).unsqueeze(0)                 # (1, 1, 96, 96, 96)

        return tensor

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _mock_prediction(filename: str) -> dict:
    """
    Generate realistic mock predictions for demo/development.
    Produces deterministic results based on filename so demo is consistent.
    """
    # Seed based on filename for consistency across requests
    seed = sum(ord(c) for c in filename) % 100
    rng  = np.random.default_rng(seed)

    # Generate realistic softmax-like probabilities
    raw    = rng.dirichlet(alpha=[2.0, 1.5, 1.5])  # biased toward AD for drama
    probs  = raw.tolist()
    pred_idx = int(np.argmax(probs))

    # Mock 256-d L2-normalised embedding
    emb  = rng.standard_normal(256).astype(np.float32)
    emb  = emb / np.linalg.norm(emb)

    return {
        "probs":    probs,
        "pred_idx": pred_idx,
        "embedding": emb,
    }


def _real_prediction(file_bytes: bytes) -> dict:
    """Run actual model inference on NIfTI bytes."""
    import torch
    import torch.nn.functional as F

    model, device = _load_model()

    tensor = _preprocess_nifti(file_bytes).to(device)

    with torch.no_grad():
        z_img, logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        emb   = z_img.squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))

    return {
        "probs":     probs.tolist(),
        "pred_idx":  pred_idx,
        "embedding": emb,
    }


def run_mri_inference(file_bytes: bytes, filename: str) -> dict:
    """
    Main inference entry point called by the API router.

    Returns a dict with all fields needed to build MRIAnalysisResponse.
    """
    from backend.models.schemas import PREDICTION_LABELS, RISK_MAP

    start = time.time()

    if _is_model_available():
        result = _real_prediction(file_bytes)
        model_version = "VisionEncoder-ResNet18-v1.0 (trained)"
    else:
        result = _mock_prediction(filename)
        model_version = "VisionEncoder-ResNet18-v1.0 (demo mode)"

    elapsed_ms = (time.time() - start) * 1000

    probs    = result["probs"]
    pred_idx = result["pred_idx"]
    emb      = result["embedding"]
    pred_cls = CLASS_LABELS[pred_idx]

    # Build per-class confidence breakdown
    severity_map = {0: 2, 1: 1, 2: 0}   # AD=2, MCI=1, CN=0
    class_scores = [
        {
            "label":      CLASS_LABELS[i],
            "confidence": float(probs[i]),
            "severity":   severity_map[i],
        }
        for i in range(NUM_CLASSES)
    ]
    # Sort by confidence descending
    class_scores.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "prediction":       pred_cls,
        "prediction_full":  PREDICTION_LABELS.get(pred_cls, pred_cls),
        "confidence":       float(probs[pred_idx]) * 100,
        "risk_level":       RISK_MAP.get(pred_cls, "UNKNOWN"),
        "class_scores":     class_scores,
        "embedding_dim":    256,
        "embedding_norm":   float(np.linalg.norm(emb)),
        "filename":         filename,
        "modality":         "Brain MRI",
        "model_version":    model_version,
        "processing_time_ms": elapsed_ms,
        "disclaimer": (
            "This output is a computational feature representation for "
            "neurological risk analysis only. It is NOT a medical diagnosis. "
            "Always consult a qualified neurologist for clinical decisions."
        ),
    }