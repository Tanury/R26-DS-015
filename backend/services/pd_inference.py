"""
backend/services/pd_inference.py

Inference service for the Parkinson's Disease Vision Encoder branch.
Supports MRI and DaTscan modalities, mock and real modes.
Includes pipeline-style response for DaTscan (step-by-step).
"""

import time
import sys
import tempfile
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
USE_MOCK_MRI = True    # set False once mri_best_model.pth is downloaded
USE_MOCK_DAT = False   # dat_best_model.pth is trained and ready

MRI_CHECKPOINT   = "mri/parkinsons/models/checkpoints/mri_best_model.pth"
DAT_CHECKPOINT   = "mri/parkinsons/models/checkpoints/dat_best_model.pth"
MRI_TARGET_SHAPE = (197, 233, 189)
DAT_RESIZE_DIM   = 224
EMBED_DIM        = 256
CLASS_LABELS     = ["HC", "PD"]

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Availability checks ────────────────────────────────────────────────────────
def _is_mri_available() -> bool:
    return Path(MRI_CHECKPOINT).exists() and not USE_MOCK_MRI

def _is_dat_available() -> bool:
    return Path(DAT_CHECKPOINT).exists() and not USE_MOCK_DAT


# ── Device ────────────────────────────────────────────────────────────────────
def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model loaders ──────────────────────────────────────────────────────────────
def _load_mri_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock3D(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm3d(out_ch)
            self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm3d(out_ch)
            self.down  = None
            if stride != 1 or in_ch != out_ch:
                self.down = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_ch)
                )
        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            if self.down: identity = self.down(x)
            return F.relu(out + identity, inplace=True)

    class ResNet3D18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem   = nn.Sequential(
                nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                nn.MaxPool3d(3, stride=2, padding=1)
            )
            self.layer1 = self._make(64,  64,  2, 1)
            self.layer2 = self._make(64,  128, 2, 2)
            self.layer3 = self._make(128, 256, 2, 2)
            self.layer4 = self._make(256, 512, 2, 2)
            self.pool   = nn.AdaptiveAvgPool3d((1, 1, 1))
        def _make(self, in_ch, out_ch, blocks, stride):
            layers = [BasicBlock3D(in_ch, out_ch, stride)]
            for _ in range(1, blocks):
                layers.append(BasicBlock3D(out_ch, out_ch, 1))
            return nn.Sequential(*layers)
        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x); x = self.layer2(x)
            x = self.layer3(x); x = self.layer4(x)
            return self.pool(x).flatten(1)

    class MRIEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = ResNet3D18()
            self.projector  = nn.Sequential(
                nn.Linear(512, EMBED_DIM), nn.BatchNorm1d(EMBED_DIM),
                nn.ReLU(inplace=True), nn.Linear(EMBED_DIM, EMBED_DIM)
            )
            self.classifier = nn.Linear(EMBED_DIM, 2)
        def forward(self, x):
            feat   = self.backbone(x)
            embed  = self.projector(feat)
            z_img  = F.normalize(embed, p=2, dim=1)
            logits = self.classifier(z_img)
            return logits, z_img

    device = _get_device()
    model  = MRIEncoder().to(device)
    ckpt   = torch.load(MRI_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def _load_dat_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBnRelu(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            )
        def forward(self, x): return self.block(x)

    class ResBlock2D(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                ConvBnRelu(channels, channels),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels)
            )
        def forward(self, x): return F.relu(x + self.block(x), inplace=True)

    class DaTCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem   = nn.Sequential(
                ConvBnRelu(1, 32, kernel=7, stride=2, padding=3),
                ConvBnRelu(32, 64, stride=2),
            )
            self.stage1 = nn.Sequential(ResBlock2D(64),  ConvBnRelu(64,  128, stride=2))
            self.stage2 = nn.Sequential(ResBlock2D(128), ConvBnRelu(128, 256, stride=2))
            self.stage3 = nn.Sequential(ResBlock2D(256), ConvBnRelu(256, 256, stride=2))
            self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, x):
            x = self.stem(x); x = self.stage1(x)
            x = self.stage2(x); x = self.stage3(x)
            return self.pool(x).flatten(1)

    class DaTEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = DaTCNN()
            self.projector  = nn.Sequential(
                nn.Linear(EMBED_DIM, EMBED_DIM), nn.BatchNorm1d(EMBED_DIM),
                nn.ReLU(inplace=True), nn.Linear(EMBED_DIM, EMBED_DIM)
            )
            self.classifier = nn.Linear(EMBED_DIM, 2)
        def forward(self, x):
            feat   = self.backbone(x)
            embed  = self.projector(feat)
            z_img  = F.normalize(embed, p=2, dim=1)
            logits = self.classifier(z_img)
            return logits, z_img

    device = _get_device()
    model  = DaTEncoder().to(device)
    ckpt   = torch.load(DAT_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


# ── Preprocessing ──────────────────────────────────────────────────────────────
def _preprocess_mri(file_bytes: bytes):
    import torch
    import nibabel as nib

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        img  = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        result = np.zeros(MRI_TARGET_SHAPE, dtype=np.float32)
        slices_src, slices_dst = [], []
        for s, t in zip(data.shape, MRI_TARGET_SHAPE):
            if s >= t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(0, t))
            else:
                pad = (t - s) // 2
                slices_src.append(slice(0, s))
                slices_dst.append(slice(pad, pad + s))
        result[tuple(slices_dst)] = data[tuple(slices_src)]
        return torch.from_numpy(result).unsqueeze(0).unsqueeze(0)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _preprocess_dat(file_bytes: bytes):
    """Full DaTscan preprocessing — returns (tensor, step_timings)."""
    import torch
    import subprocess
    import nibabel as nib
    import cv2

    timings = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 0 — write DICOM
        dcm_path = Path(tmpdir) / "input.dcm"
        dcm_path.write_bytes(file_bytes)

        # Step 1 — DICOM → NIfTI
        t0     = time.time()
        out_dir = Path(tmpdir) / "nifti"
        out_dir.mkdir()
        subprocess.run(
            ["dcm2niix", "-z", "y", "-f", "dat", "-o", str(out_dir), str(tmpdir)],
            capture_output=True
        )
        nii_files = list(out_dir.glob("*.nii.gz"))
        if not nii_files:
            raise ValueError("dcm2niix conversion failed — check DICOM format")
        timings["dicom_to_nifti"] = round(time.time() - t0, 2)

        # Step 2 — Slice extraction
        t0   = time.time()
        img  = nib.load(str(nii_files[0]))
        data = img.get_fdata(dtype=np.float32)
        if data.shape[1] > 120:
            data = data[:, :120, :]
        lo = int(data.shape[1] * 0.30)
        hi = int(data.shape[1] * 0.70)
        slice_means = data[:, lo:hi, :].mean(axis=(0, 2))
        best_local  = int(np.argmax(slice_means))
        best_ax1    = lo + best_local
        w_lo        = max(0, best_ax1 - 4)
        w_hi        = min(data.shape[1], best_ax1 + 5)
        slice_2d    = data[:, w_lo:w_hi, :].mean(axis=1)
        timings["slice_extraction"] = round(time.time() - t0, 2)

        # Step 3 — Resize + SBR normalise
        t0      = time.time()
        resized = cv2.resize(slice_2d, (DAT_RESIZE_DIM, DAT_RESIZE_DIM),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32)
        h          = resized.shape[0]
        ref_region = resized[int(h * 0.75):, :]
        ref_vals   = ref_region[ref_region > 0]
        if len(ref_vals) > 10:
            resized = resized / (ref_vals.mean() + 1e-8)
        else:
            nonzero = resized[resized > 0]
            p_low, p_high = np.percentile(nonzero, 1), np.percentile(nonzero, 99)
            resized = np.clip(resized, p_low, p_high)
            if p_high > p_low:
                resized = (resized - p_low) / (p_high - p_low)
        timings["sbr_normalise"] = round(time.time() - t0, 2)

        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
        return tensor, timings


# ── Mock predictions ───────────────────────────────────────────────────────────
def _mock_prediction(filename: str, seed_offset: int = 0) -> dict:
    seed = (sum(ord(c) for c in filename) + seed_offset) % 100
    rng  = np.random.default_rng(seed)
    raw  = rng.dirichlet(alpha=[1.5, 2.0])
    probs = raw.tolist()
    emb   = rng.standard_normal(EMBED_DIM).astype(np.float32)
    emb   = emb / np.linalg.norm(emb)
    return {"probs": probs, "pred_idx": int(np.argmax(probs)), "embedding": emb}


# ── Real predictions ───────────────────────────────────────────────────────────
def _real_mri_prediction(file_bytes: bytes) -> dict:
    import torch
    import torch.nn.functional as F
    model, device = _load_mri_model()
    tensor = _preprocess_mri(file_bytes).to(device)
    with torch.no_grad():
        logits, z_img = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        emb   = z_img.squeeze().cpu().numpy()
    return {"probs": probs.tolist(), "pred_idx": int(np.argmax(probs)), "embedding": emb}


def _real_dat_prediction(file_bytes: bytes) -> tuple:
    """Returns (result_dict, timings_dict)."""
    import torch
    import torch.nn.functional as F
    model, device = _load_dat_model()
    t0 = time.time()
    tensor, timings = _preprocess_dat(file_bytes)
    tensor = tensor.to(device)
    t_infer = time.time()
    with torch.no_grad():
        logits, z_img = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        emb   = z_img.squeeze().cpu().numpy()
    timings["inference"] = round(time.time() - t_infer, 2)
    return {"probs": probs.tolist(), "pred_idx": int(np.argmax(probs)), "embedding": emb}, timings


# ── Response builder ───────────────────────────────────────────────────────────
def _build_response(result: dict, filename: str,
                    modality: str, model_version: str) -> dict:
    from backend.models.schemas import PREDICTION_LABELS, RISK_MAP
    probs    = result["probs"]
    pred_idx = result["pred_idx"]
    emb      = result["embedding"]
    pred_cls = CLASS_LABELS[pred_idx]
    class_scores = [
        {"label": CLASS_LABELS[i], "confidence": float(probs[i]),
         "severity": 1 if CLASS_LABELS[i] == "PD" else 0}
        for i in range(len(CLASS_LABELS))
    ]
    class_scores.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "prediction":         pred_cls,
        "prediction_full":    PREDICTION_LABELS.get(pred_cls, pred_cls),
        "confidence":         float(probs[pred_idx]) * 100,
        "risk_level":         RISK_MAP.get(pred_cls, "UNKNOWN"),
        "class_scores":       class_scores,
        "embedding_dim":      EMBED_DIM,
        "embedding_norm":     float(np.linalg.norm(emb)),
        "filename":           filename,
        "modality":           modality,
        "model_version":      model_version,
        "processing_time_ms": 0.0,
        "disclaimer": (
            "This output is a computational feature representation for "
            "neurological risk analysis only. It is NOT a medical diagnosis. "
            "Always consult a qualified neurologist for clinical decisions."
        ),
    }


# ── Main entry points ──────────────────────────────────────────────────────────
def run_mri_inference(file_bytes: bytes, filename: str) -> dict:
    start = time.time()
    if _is_mri_available():
        result        = _real_mri_prediction(file_bytes)
        model_version = "PD-MRIEncoder-ResNet3D-v1.0 (trained)"
    else:
        result        = _mock_prediction(filename, seed_offset=0)
        model_version = "PD-MRIEncoder-ResNet3D-v1.0 (demo mode)"
    response = _build_response(result, filename, "Brain MRI", model_version)
    response["processing_time_ms"] = (time.time() - start) * 1000
    return response


def run_dat_inference(file_bytes: bytes, filename: str) -> dict:
    """Simple single-result DaTscan inference (used by /pd/analyze/dat)."""
    start = time.time()
    if _is_dat_available():
        result, _     = _real_dat_prediction(file_bytes)
        model_version = "PD-DaTEncoder-2DCNN-v1.0 (trained)"
    else:
        result        = _mock_prediction(filename, seed_offset=42)
        model_version = "PD-DaTEncoder-2DCNN-v1.0 (demo mode)"
    response = _build_response(result, filename, "DaTscan SPECT", model_version)
    response["processing_time_ms"] = (time.time() - start) * 1000
    return response


def run_dat_pipeline(file_bytes: bytes, filename: str) -> dict:
    """
    Pipeline-style DaTscan inference with step-by-step breakdown.
    Used by /pd/analyze/dat/pipeline — matches the AD pipeline response format.
    """
    start_total  = time.time()
    pipeline     = []
    model_version = "PD-DaTEncoder-2DCNN-v1.0 (trained)" if _is_dat_available() \
                    else "PD-DaTEncoder-2DCNN-v1.0 (demo mode)"

    if _is_dat_available():
        # Real inference with per-step timing
        try:
            t0 = time.time()
            result, timings = _real_dat_prediction(file_bytes)

            pipeline.append({
                "step": 0, "name": "DICOM → NIfTI",
                "description": "dcm2niix conversion · Reconstructed DaTscan SPECT volume · Multi-frame DICOM handling",
                "success": True, "elapsed_s": timings.get("dicom_to_nifti", 0),
                "slices": None, "error": None,
            })
            pipeline.append({
                "step": 1, "name": "Slice Extraction",
                "description": f"Peak transaxial slice ±4 averaged · Axis 1 (109 slices) · Striatum region identified",
                "success": True, "elapsed_s": timings.get("slice_extraction", 0),
                "slices": None, "error": None,
            })
            pipeline.append({
                "step": 2, "name": "SBR Normalisation",
                "description": "Specific Binding Ratio · Occipital reference region · Resize 224×224 · Float32",
                "success": True, "elapsed_s": timings.get("sbr_normalise", 0),
                "slices": None, "error": None,
            })
            pipeline.append({
                "step": 3, "name": "2D-CNN Inference",
                "description": "DaTEncoder · 2D-CNN backbone · 256-d L2-normalised z_img · PD/HC classification",
                "success": True, "elapsed_s": timings.get("inference", 0),
                "slices": None, "error": None,
            })

            prediction = _build_response(result, filename, "DaTscan SPECT", model_version)
            prediction["processing_time_ms"] = (time.time() - start_total) * 1000

        except Exception as e:
            pipeline.append({
                "step": 0, "name": "DICOM → NIfTI",
                "description": "Conversion failed",
                "success": False, "elapsed_s": round(time.time() - start_total, 2),
                "slices": None, "error": str(e),
            })
            return {
                "pipeline_success": False,
                "pipeline": pipeline,
                "total_elapsed_s": round(time.time() - start_total, 2),
                "subject_id": filename,
                "prediction": {"error": str(e)},
            }
    else:
        # Mock mode — simulate step timings
        import random
        mock_timings = [0.8, 0.3, 0.2, 0.4]
        step_info = [
            ("DICOM → NIfTI",    "dcm2niix conversion · Reconstructed DaTscan SPECT · Demo mode"),
            ("Slice Extraction", "Peak transaxial slice ±4 averaged · Axis 1 · Striatum identified · Demo mode"),
            ("SBR Normalisation","Specific Binding Ratio · Occipital reference · 224×224 · Demo mode"),
            ("2D-CNN Inference", "DaTEncoder · 256-d z_img · PD/HC · Demo mode"),
        ]
        for i, (name, desc) in enumerate(step_info):
            time.sleep(mock_timings[i] * 0.1)  # brief pause for realism
            pipeline.append({
                "step": i, "name": name, "description": desc,
                "success": True, "elapsed_s": mock_timings[i],
                "slices": None, "error": None,
            })

        result     = _mock_prediction(filename, seed_offset=42)
        prediction = _build_response(result, filename, "DaTscan SPECT", model_version)
        prediction["processing_time_ms"] = (time.time() - start_total) * 1000

    return {
        "pipeline_success": True,
        "pipeline":         pipeline,
        "total_elapsed_s":  round(time.time() - start_total, 2),
        "subject_id":       filename,
        "prediction":       prediction,
    }