import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import (
    BiomedicalRecordExtractionError,
    ModelLoadError,
    PredictionError,
)
from app.schemas.biomedical_record import BiomedicalRecordExtractionResponse
from app.schemas.neurological_risk_request import NeurologicalRiskRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.gemini_biomedical_service import extract_biomedical_record
from app.services.neurological_prediction_service import (
    predict_neurological_risk,
)


logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_RECORD_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}


def _has_record_signature(data: bytes, mime_type: str) -> bool:
    if mime_type == "application/pdf":
        return data.startswith(b"%PDF-")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/webp":
        return len(data) > 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


@router.post(
    "/extract-record",
    response_model=BiomedicalRecordExtractionResponse,
    summary="Extract biomedical form values from a document or image",
)
async def extract_record(
    file: Annotated[UploadFile, File(...)],
) -> BiomedicalRecordExtractionResponse:
    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_RECORD_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file format. Upload a PDF, JPG, PNG, or WebP file.",
        )

    document_bytes = await file.read(settings.max_biomedical_record_bytes + 1)
    filename = Path(file.filename or "biomedical-record").name
    await file.close()
    if not document_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(document_bytes) > settings.max_biomedical_record_bytes:
        raise HTTPException(
            status_code=413,
            detail="Biomedical record files must be 15 MB or smaller.",
        )
    if not _has_record_signature(document_bytes, mime_type):
        raise HTTPException(
            status_code=422,
            detail="The file content does not match its declared format.",
        )

    try:
        analysis = await run_in_threadpool(
            extract_biomedical_record,
            document_bytes,
            mime_type,
        )
    except BiomedicalRecordExtractionError as exc:
        logger.warning("biomedical_record_rejected reason=%s", exc)
        status = 503 if "configured" in str(exc) or "package" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    extracted_count = sum(
        value is not None for value in analysis.features.model_dump().values()
    )
    return BiomedicalRecordExtractionResponse(
        filename=filename,
        extraction_quality=analysis.extraction_quality,
        quality_notes=analysis.quality_notes,
        extracted_count=extracted_count,
        features=analysis.features,
        extraction_disclaimer=(
            "Gemini transcribed these values from the uploaded record. Review every "
            "field against the source before assessment; this is not a diagnosis."
        ),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict neurological disease pattern and biomedical risk",
)
def predict(
    payload: NeurologicalRiskRequest,
) -> PredictionResponse:
    try:
        return predict_neurological_risk(payload)
    except ModelLoadError as exc:
        logger.exception("Neurological prediction model is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Neurological prediction model is temporarily unavailable.",
        ) from exc
    except PredictionError as exc:
        logger.exception("Neurological prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Unable to complete neurological prediction.",
        ) from exc
