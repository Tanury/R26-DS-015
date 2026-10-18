import json
import logging
from typing import Any

from pydantic import ValidationError

from app.core.config import settings
from app.core.exceptions import AudioFeatureExtractionError
from app.schemas.voice_assessment import GeminiAudioAnalysis
from app.services.audio_preparation import prepare_browser_audio


logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
Analyze this single-speaker speech recording as a research acoustic screening input.
Do not diagnose a disease or infer a neurological condition. Return only the
structured output requested by the schema.

Estimate these acoustic values using the exact units below:
- mfcc_1_mean, mfcc_2_mean, mfcc_3_mean: mean MFCC coefficients
- pitch_mean and pitch_std: Hz
- jitter and shimmer: local percentage values
- hnr: dB
- speech_rate: spoken words per second
- pause_count: number of meaningful silent pauses
- mean_pause_duration: seconds
- mean_energy: mean signal intensity in dB
- spectral_centroid_mean: Hz
- zero_crossing_rate_mean: ratio from 0 to 1

Set quality_status to unusable for silence, music-only audio, multiple overlapping
speakers, severe clipping, or audio too unclear for speech analysis. Use limited when
the estimates are possible but recording quality is weak. Include concise quality
notes and a short transcript when speech is intelligible. Never invent patient facts.
Natural pauses, an accent, non-English speech, or mild background noise alone are
not reasons to reject audible single-speaker speech. Use limited if meaningful
estimates can still be made. For sustained_vowel, a held vowel is the expected
input; no words or transcript are required. Do not require a reading passage for
monologue or picture_description. For unusable audio, explain the specific audible
problem in quality_notes so the speaker can correct it.
""".strip()


def _gemini_response_schema() -> dict[str, Any]:
    """Return the strict response model in Gemini's supported schema subset.

    ``extra="forbid"`` makes Pydantic emit ``additionalProperties: false`` for
    each object.  The Gemini Developer API's legacy ``response_schema`` field
    rejects that otherwise-valid JSON Schema keyword.  Remove it only from the
    wire schema; ``GeminiAudioAnalysis.model_validate`` below still enforces the
    backend's strict no-extra-fields contract on every response.
    """
    schema = GeminiAudioAnalysis.model_json_schema()

    def remove_unsupported_fields(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for value in node.values():
                remove_unsupported_fields(value)
        elif isinstance(node, list):
            for value in node:
                remove_unsupported_fields(value)

    remove_unsupported_fields(schema)
    return schema


def _describe_gemini_failure(exc: Exception) -> str:
    """Turn a Gemini SDK error into an actionable message without leaking secrets."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    message = str(getattr(exc, "message", "") or "")
    if "API key" in message or "API_KEY" in str(exc):
        return (
            "Gemini rejected the server's API key. Set a valid GEMINI_API_KEY in "
            "backend/.env and restart the backend."
        )
    if code == 403 or status == "PERMISSION_DENIED":
        return "The Gemini API key does not have permission to use this model."
    if code == 404 or status == "NOT_FOUND":
        return (
            f"Gemini model '{settings.gemini_model}' is not available. "
            "Check GEMINI_MODEL in backend/.env."
        )
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return "Gemini quota or rate limit reached. Wait a minute and try again."
    if isinstance(code, int) and code >= 500:
        return "Gemini is temporarily unavailable. Try again shortly."
    return "Gemini could not analyze the audio sample."


def extract_speech_features(
    audio_bytes: bytes, mime_type: str, recording_task: str = "reading"
) -> GeminiAudioAnalysis:
    api_key = settings.gemini_api_key.strip()
    if not api_key or api_key.lower().startswith(("replace", "your", "<")):
        raise AudioFeatureExtractionError(
            "GEMINI_API_KEY is not configured on the server. Set a real key in "
            "backend/.env and restart the backend."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise AudioFeatureExtractionError(
            "The google-genai package is required for voice analysis."
        ) from exc

    audio_bytes, mime_type = prepare_browser_audio(audio_bytes, mime_type)

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                f"{EXTRACTION_PROMPT}\n\nSelected recording task: {recording_task}.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_gemini_response_schema(),
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        if not response.text:
            raise AudioFeatureExtractionError("Gemini returned an empty analysis.")
        analysis = GeminiAudioAnalysis.model_validate(json.loads(response.text))
    except AudioFeatureExtractionError:
        raise
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.exception("gemini_audio_response_invalid")
        raise AudioFeatureExtractionError(
            "Gemini returned an invalid acoustic feature response."
        ) from exc
    except Exception as exc:
        logger.exception("gemini_audio_request_failed")
        raise AudioFeatureExtractionError(_describe_gemini_failure(exc)) from exc

    if analysis.quality_status == "unusable":
        notes = " ".join(note.strip() for note in analysis.quality_notes if note.strip())[:600]
        raise AudioFeatureExtractionError(
            "Speech extraction could not use this recording. "
            + (f"Audio quality feedback: {notes} " if notes else "No reliable speech features could be extracted. ")
            + "Listen to the preview, check the microphone level, and record the selected task again in a quiet room."
        )

    logger.info(
        "gemini_audio_extraction_complete quality=%s model=%s",
        analysis.quality_status,
        settings.gemini_model,
    )
    return analysis
