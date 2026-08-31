"""
backend/routers/analyze_ms.py

API router for Multiple Sclerosis analysis endpoints.

Endpoints:
    POST /api/ms/analyze/oct    — .vol + .mat -> MS risk (retinal thickness branch)
    POST /api/ms/analyze/mri    — NIfTI -> MS risk (MRI branch) -- NOT YET IMPLEMENTED
    GET  /api/ms/model/status   — OCT + MRI model status
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.ms_schemas import MSAnalysisResponse, MSModelStatusResponse
from backend.services.ms_inference import (
    run_oct_inference,
    _is_oct_available,
    #_is_mri_available,
    OCT_CHECKPOINT,
)
from backend.services.ms_mri_inference import (
    run_mri_inference,
    _is_mri_available,
    MRI_CHECKPOINT,
)

router = APIRouter()

ALLOWED_VOL_EXTENSIONS = {".vol"}
ALLOWED_MAT_EXTENSIONS = {".mat"}
ALLOWED_MRI_EXTENSIONS = {".nii", ".nii.gz"}
MAX_OCT_FILE_SIZE_MB = 250  # .vol files run large (~100MB for a 49-B-scan Spectralis cube)
MAX_MRI_FILE_SIZE_MB = 500  


def _validate_file(file: UploadFile, allowed_exts: set) -> None:
    filename = file.filename or "unknown"
    if not any(filename.endswith(e) for e in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{filename}'. Expected: {', '.join(allowed_exts)}"
        )


# ── OCT endpoint ───────────────────────────────────────────────────────────────
@router.post("/ms/analyze/oct", response_model=MSAnalysisResponse)
async def analyze_ms_oct(
    vol_file: UploadFile = File(...),
    mat_file: UploadFile = File(...),
):
    """
    Upload a Heidelberg Spectralis .vol OCT scan and its paired .mat manual
    layer delineation, and receive MS risk analysis based on central-window
    (foveal B-scan) retinal-layer thickness.

    Please upload both a .vol file and a .mat file. This branch was trained and validated on
    the manually-delineated JHU MS/HC OCT dataset and does not perform its
    own automated layer segmentation, so a .vol file alone is not sufficient
    (see project notes on this limitation).
    """
    _validate_file(vol_file, ALLOWED_VOL_EXTENSIONS)
    _validate_file(mat_file, ALLOWED_MAT_EXTENSIONS)

    vol_bytes = await vol_file.read()
    mat_bytes = await mat_file.read()
    vol_filename = vol_file.filename or "unknown.vol"
    mat_filename = mat_file.filename or "unknown.mat"

    if len(vol_bytes) / 1e6 > MAX_OCT_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f".vol file too large. Max {MAX_OCT_FILE_SIZE_MB} MB.")
    if not vol_bytes:
        raise HTTPException(status_code=400, detail="Uploaded .vol file is empty.")
    if not mat_bytes:
        raise HTTPException(status_code=400, detail="Uploaded .mat file is empty.")

    try:
        result = run_oct_inference(vol_bytes, vol_filename, mat_bytes, mat_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCT inference failed: {e}")
    return MSAnalysisResponse(**result)


# ── MRI endpoint
@router.post("/ms/analyze/mri")
async def analyze_ms_mri(file: UploadFile = File(...)):
    """
    Upload a RAW brain MRI (.nii or .nii.gz)
    No preprocessing is required.
    Runs MindGlide (pretrained, published segmentation model) to extract
    region volumes, then the trained region-volume encoder to produce an
    MS risk prediction.
    """
    _validate_file(file, ALLOWED_MRI_EXTENSIONS)
    file_bytes = await file.read()
    filename = file.filename or "unknown.nii.gz"
 
    if len(file_bytes) / 1e6 > MAX_MRI_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_MRI_FILE_SIZE_MB} MB.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
 
    try:
        result = run_mri_inference(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI inference failed: {e}")
    return MSAnalysisResponse(**result)


# ── Status endpoint ────────────────────────────────────────────────────────────
@router.get("/ms/model/status", response_model=MSModelStatusResponse)
def ms_model_status():
    oct_real = _is_oct_available()
    mri_real = _is_mri_available()

    if oct_real and mri_real:
        message = "Both OCT and MRI models loaded and running."
    elif oct_real:
        message = f"OCT model loaded from {OCT_CHECKPOINT.name}. MRI checkpoint not found."
    elif mri_real:
        message = f"MRI model loaded from {MRI_CHECKPOINT.name}. OCT checkpoint not found."
    else:
        message = "No OCT or MRI checkpoints found. " #run the respective train_*.py scripts first

    return MSModelStatusResponse(
        oct_mode="real" if oct_real else "unavailable",
        mri_mode="real" if mri_real else "unavailable",
        oct_loaded=oct_real,
        mri_loaded=mri_real,
        message=message,
    )