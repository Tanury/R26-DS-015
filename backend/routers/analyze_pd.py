"""
backend/routers/analyze_pd.py

API router for Parkinson's Disease analysis endpoints.

Endpoints:
    POST /api/pd/analyze/mri              — NIfTI → PD risk (MRI branch)
    POST /api/pd/analyze/dat              — DICOM → PD risk (DaTscan, simple)
    POST /api/pd/analyze/dat/pipeline     — DICOM → PD risk (DaTscan, step-by-step)
    GET  /api/pd/model/status             — MRI + DaTscan model status
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.pd_schemas import PDAnalysisResponse, PDModelStatusResponse
from backend.services.pd_inference import (
    run_mri_inference,
    run_dat_inference,
    run_dat_pipeline,
    _is_mri_available,
    _is_dat_available,
    MRI_CHECKPOINT,
    DAT_CHECKPOINT,
)

router = APIRouter()

ALLOWED_MRI_EXTENSIONS = {".nii", ".nii.gz"}
ALLOWED_DAT_EXTENSIONS = {".dcm", ".ima", ".img"}
MAX_FILE_SIZE_MB        = 600


def _validate_file(file: UploadFile, allowed_exts: set) -> None:
    filename = file.filename or "unknown"
    if not any(filename.endswith(e) for e in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{filename}'. Expected: {', '.join(allowed_exts)}"
        )


# ── MRI endpoint ───────────────────────────────────────────────────────────────
@router.post("/pd/analyze/mri", response_model=PDAnalysisResponse)
async def analyze_pd_mri(file: UploadFile = File(...)):
    """
    Upload a preprocessed T1-weighted MRI NIfTI and receive PD risk analysis.
    Note: MRI branch is supplementary — DaTscan is the primary modality for PD.
    """
    _validate_file(file, ALLOWED_MRI_EXTENSIONS)
    file_bytes = await file.read()
    filename   = file.filename or "unknown.nii.gz"

    if len(file_bytes) / 1e6 > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB} MB.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = run_mri_inference(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI inference failed: {e}")
    return PDAnalysisResponse(**result)


# ── DaTscan simple endpoint ────────────────────────────────────────────────────
@router.post("/pd/analyze/dat", response_model=PDAnalysisResponse)
async def analyze_pd_dat(file: UploadFile = File(...)):
    """
    Upload a reconstructed DaTscan SPECT DICOM and receive PD risk analysis.
    Returns a single result — use /pd/analyze/dat/pipeline for step-by-step view.
    """
    _validate_file(file, ALLOWED_DAT_EXTENSIONS)
    file_bytes = await file.read()
    filename   = file.filename or "unknown.dcm"

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = run_dat_inference(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DaTscan inference failed: {e}")
    return PDAnalysisResponse(**result)


# ── DaTscan pipeline endpoint ─────────────────────────────────────────────────
@router.post("/pd/analyze/dat/pipeline")
async def analyze_pd_dat_pipeline(file: UploadFile = File(...)):
    """
    Upload a reconstructed DaTscan SPECT DICOM.
    Returns step-by-step pipeline response matching the AD preprocessing format:
      Step 0: DICOM → NIfTI
      Step 1: Slice Extraction
      Step 2: SBR Normalisation
      Step 3: 2D-CNN Inference
    """
    _validate_file(file, ALLOWED_DAT_EXTENSIONS)
    file_bytes = await file.read()
    filename   = file.filename or "unknown.dcm"

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = run_dat_pipeline(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DaTscan pipeline failed: {e}")
    return result


# ── Status endpoint ────────────────────────────────────────────────────────────
@router.get("/pd/model/status", response_model=PDModelStatusResponse)
def pd_model_status():
    mri_real = _is_mri_available()
    dat_real = _is_dat_available()
    if mri_real and dat_real:
        message = "Both MRI and DaTscan models loaded and running."
    elif dat_real:
        message = "DaTscan model loaded (AUC 93.75%). MRI in demo mode."
    elif mri_real:
        message = "MRI model loaded. DaTscan in demo mode."
    else:
        message = "Both models in demo mode — mock predictions active."
    return PDModelStatusResponse(
        mri_mode=  "real" if mri_real else "mock",
        dat_mode=  "real" if dat_real else "mock",
        mri_loaded= mri_real,
        dat_loaded= dat_real,
        message=    message,
    )