import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import (
    AudioFeatureExtractionError,
    FeatureValidationError,
    ModelLoadError,
    PredictionError,
)
from app.schemas.prediction_request import PredictionRequest
from app.schemas.voice_assessment import VoiceAssessmentResponse
from app.services.gemini_audio_service import extract_speech_features
from app.services.prediction_service import predict_risk


logger = logging.getLogger(__name__)
router = APIRouter()

RecordingTask = Literal["reading", "monologue", "picture_description", "sustained_vowel"]
ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
}
MIME_ALIASES = {"audio/x-wav": "audio/wav", "audio/x-m4a": "audio/mp4"}


def _has_audio_signature(data: bytes, mime_type: str) -> bool:
    if mime_type in {"audio/wav", "audio/x-wav"}:
        return len(data) > 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if mime_type == "audio/mpeg":
        return data.startswith(b"ID3") or (
            len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
        )
    if mime_type in {"audio/mp4", "audio/x-m4a"}:
        return len(data) > 12 and data[4:8] == b"ftyp"
    if mime_type == "audio/ogg":
        return data.startswith(b"OggS")
    if mime_type == "audio/webm":
        return data.startswith(b"\x1a\x45\xdf\xa3")
    return False


@router.post("/", response_model=VoiceAssessmentResponse)
async def assess_voice(
    file: Annotated[UploadFile, File(...)],
    patient_age: Annotated[int, Form(ge=18, le=120)],
    recording_task: Annotated[RecordingTask, Form()],
) -> VoiceAssessmentResponse:
    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")

    audio_bytes = await file.read(settings.max_audio_bytes + 1)
    filename = Path(file.filename or "voice-sample").name
    await file.close()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="The audio file is empty.")
    if len(audio_bytes) > settings.max_audio_bytes:
        raise HTTPException(status_code=413, detail="Audio files must be 18 MB or smaller.")
    if not _has_audio_signature(audio_bytes, mime_type):
        raise HTTPException(status_code=422, detail="The file content is not valid audio.")

    try:
        analysis = await run_in_threadpool(
            extract_speech_features,
            audio_bytes,
            MIME_ALIASES.get(mime_type, mime_type),
        )
        prediction = await run_in_threadpool(
            predict_risk,
            PredictionRequest(features=analysis.features.model_dump()),
        )
    except AudioFeatureExtractionError as exc:
        logger.warning("voice_assessment_rejected reason=%s", exc)
        status = 503 if "configured" in str(exc) or "package" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelLoadError as exc:
        logger.exception("voice_prediction_model_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Prediction model is temporarily unavailable.",
        ) from exc
    except PredictionError as exc:
        logger.exception("voice_prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from exc

    return VoiceAssessmentResponse(
        filename=filename,
        recording_task=recording_task,
        patient_age=patient_age,
        extraction_quality=analysis.quality_status,
        quality_notes=analysis.quality_notes,
        transcript=analysis.transcript,
        extracted_features=analysis.features,
        prediction=prediction,
        extraction_disclaimer=(
            "Gemini-derived acoustic values are structured research estimates, not "
            "laboratory-grade signal measurements or a medical diagnosis."
        ),
    )
