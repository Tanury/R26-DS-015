"""
preprocess.py
=============
API router for full DICOM → preprocessing → inference pipeline.

Endpoints:
    POST /api/preprocess/dicom   — Upload DICOM zip, get full pipeline + prediction
    POST /api/preprocess/status  — Check pipeline progress (future use)
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.services.preprocessing_pipeline import run_full_pipeline
from backend.services.inference import run_mri_inference, _is_model_available

router = APIRouter()

MAX_ZIP_SIZE_MB = 2000


@router.post("/preprocess/dicom")
async def preprocess_dicom(file: UploadFile = File(...)):
    """
    Upload a DICOM zip file.
    Runs full 4-step preprocessing pipeline then inference.

    Returns:
    - steps: list of preprocessing steps with axial/coronal/sagittal slice images
    - prediction: final classification result (AD/MCI/CN)
    - pipeline_complete: bool
    """
    filename = file.filename or "scan.zip"

    if not filename.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .zip file containing the DICOM folder."
        )

    zip_bytes = await file.read()

    size_mb = len(zip_bytes) / (1024 * 1024)
    if size_mb > MAX_ZIP_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.0f} MB). Maximum is {MAX_ZIP_SIZE_MB} MB."
        )

    if len(zip_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run preprocessing pipeline
    pipeline_result = run_full_pipeline(zip_bytes, filename)

    # Run inference on denoised output if pipeline succeeded
    prediction = None
    if pipeline_result["success"] and pipeline_result["final_nifti"]:
        try:
            final_path = pipeline_result["final_nifti"]
            with open(final_path, "rb") as f:
                nifti_bytes = f.read()
            prediction = run_mri_inference(
                nifti_bytes,
                Path(final_path).name
            )
        except Exception as e:
            prediction = {
                "error": str(e),
                "prediction": "ERROR",
                "confidence": 0,
                "risk_level": "UNKNOWN",
                "class_scores": [],
                "disclaimer": "Inference failed after preprocessing.",
            }

    return JSONResponse({
        "pipeline":          pipeline_result["steps"],
        "pipeline_success":  pipeline_result["success"],
        "pipeline_error":    pipeline_result.get("error"),
        "total_elapsed_s":   pipeline_result["total_elapsed_s"],
        "subject_id":        pipeline_result["subject_id"],
        "prediction":        prediction,
        "model_mode":        "real" if _is_model_available() else "mock",
    })


@router.get("/preprocess/check")
def check_tools():
    """Check if required preprocessing tools are available."""
    import subprocess

    tools = {}
    for tool in ["dcm2niix", "flirt", "bet", "fslreorient2std"]:
        r = subprocess.run(["which", tool], capture_output=True, text=True)
        tools[tool] = {
            "available": r.returncode == 0,
            "path": r.stdout.strip() if r.returncode == 0 else None,
        }

    try:
        import ants
        tools["antspyx"] = {"available": True, "version": ants.__version__}
    except ImportError:
        tools["antspyx"] = {"available": False, "version": None}

    all_ok = all(v["available"] for v in tools.values())

    return {
        "tools": tools,
        "all_available": all_ok,
        "message": "All tools ready" if all_ok else "Some tools missing — check FSL and ANTs installation",
    }