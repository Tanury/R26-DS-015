"""
backend/routers/analyze.py
API router for MRI and retinal image analysis endpoints.

Endpoints:
    POST /api/analyze/mri      — Upload NIfTI file, get risk analysis
    GET  /api/model/status     — Check if real model is loaded or mock mode
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.schemas import MRIAnalysisResponse
from backend.services.inference import run_mri_inference, _is_model_available

router = APIRouter()

ALLOWED_EXTENSIONS = {".nii", ".nii.gz", ".gz"}
MAX_FILE_SIZE_MB   = 500


@router.post("/analyze/mri", response_model=MRIAnalysisResponse)
async def analyze_mri(file: UploadFile = File(...)):
    """
    Upload a preprocessed NIfTI MRI file and receive neurological risk analysis.

    Input:  .nii or .nii.gz file (T1-weighted brain MRI, preprocessed)
    Output: Predicted class, confidence scores, z_img embedding metadata

    Note: File should be preprocessed (registered + skull-stripped +
    bias-corrected) for best results. Raw DICOM/NIfTI will produce
    less reliable predictions.
    """
    # Validate filename
    filename = file.filename or "unknown.nii.gz"
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type: '{filename}'. "
                "Please upload a NIfTI file (.nii or .nii.gz)."
            ),
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run inference
    try:
        result = run_mri_inference(file_bytes, filename)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )

    return MRIAnalysisResponse(**result)


@router.get("/model/status")
def model_status():
    """Returns whether real model or mock mode is active."""
    real = _is_model_available()
    return {
        "mode":        "real" if real else "mock",
        "checkpoint":  "mri/alzheimers/models/checkpoints/best_model.pth",
        "loaded":      real,
        "message": (
            "Real VisionEncoder model loaded and running."
            if real else
            "Demo mode — mock predictions active. "
            "Place best_model.pth in checkpoints/ to enable real inference."
        ),
    }


@router.get("/diseases")
def diseases():
    """Returns the list of supported diseases and their status."""
    return {
        "diseases": [
            {
                "id":       "alzheimers",
                "name":     "Alzheimer's Disease",
                "modality": "Brain MRI",
                "classes":  ["AD", "MCI", "CN"],
                "status":   "active",
            },
            {
                "id":       "parkinsons",
                "name":     "Parkinson's Disease",
                "modality": "DaTscan SPECT + MRI",
                "classes":  ["PD", "HC"],
                "status":   "active",
            },
            {
                "id":       "multiple_sclerosis",
                "name":     "Multiple Sclerosis",
                "modality": "OCT Retinal Images",
                "classes":  ["MS", "HC"],
                "status":   "active",
            },
        ]
    }