import json
import logging

from pydantic import ValidationError

from app.core.config import settings
from app.core.exceptions import AudioFeatureExtractionError
from app.schemas.voice_assessment import GeminiAudioAnalysis


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
""".strip()


def extract_speech_features(audio_bytes: bytes, mime_type: str) -> GeminiAudioAnalysis:
    if not settings.gemini_api_key:
        raise AudioFeatureExtractionError(
            "GEMINI_API_KEY is not configured on the server."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise AudioFeatureExtractionError(
            "The google-genai package is required for voice analysis."
        ) from exc

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                EXTRACTION_PROMPT,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": GeminiAudioAnalysis.model_json_schema(),
                "temperature": 0,
            },
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
        raise AudioFeatureExtractionError(
            "Gemini could not analyze the audio sample."
        ) from exc

    if analysis.quality_status == "unusable":
        raise AudioFeatureExtractionError(
            "The audio sample is not suitable for speech feature extraction."
        )

    logger.info(
        "gemini_audio_extraction_complete quality=%s model=%s",
        analysis.quality_status,
        settings.gemini_model,
    )
    return analysis
