import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.exceptions import AudioFeatureExtractionError
from app.services import gemini_audio_service as service
from app.services.gemini_audio_service import _gemini_response_schema


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_gemini_wire_schema_excludes_unsupported_additional_properties() -> None:
    schema = _gemini_response_schema()

    assert not _contains_key(schema, "additionalProperties")
    assert schema["properties"]["features"]["$ref"].endswith(
        "/ExtractedSpeechFeatures"
    )


@pytest.fixture
def fake_gemini(monkeypatch):
    from google import genai
    response = {
        "quality_status": "usable", "transcript": "Clearly spoken words.", "quality_notes": [],
        "features": {
            "mfcc_1_mean": -245.2, "mfcc_2_mean": 82.1, "mfcc_3_mean": 14.7,
            "pitch_mean": 178.4, "pitch_std": 32.8, "jitter": 0.9, "shimmer": 3.1,
            "hnr": 18.6, "speech_rate": 2.4, "pause_count": 12,
            "mean_pause_duration": 0.42, "mean_energy": 61.3,
            "spectral_centroid_mean": 1840.5, "zero_crossing_rate_mean": 0.08,
        },
    }
    generate = Mock(side_effect=lambda **kwargs: SimpleNamespace(text=json.dumps(response)))
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(service.settings, "gemini_api_key", "unit-test-key")
    prepare = Mock(return_value=(b"decoded-wav", "audio/wav"))
    monkeypatch.setattr(service, "prepare_browser_audio", prepare)
    return response, generate, prepare


@pytest.mark.parametrize("quality", ["usable", "limited"])
def test_audible_recording_proceeds_with_original_quality(fake_gemini, quality):
    response, generate, prepare = fake_gemini
    response["quality_status"] = quality
    result = service.extract_speech_features(b"webm", "audio/webm", "monologue")
    assert result.quality_status == quality
    prepare.assert_called_once_with(b"webm", "audio/webm")
    contents = generate.call_args.kwargs["contents"]
    assert "Selected recording task: monologue" in contents[0]
    assert contents[1].inline_data.mime_type == "audio/wav"
    assert contents[1].inline_data.data == b"decoded-wav"


def test_sustained_vowel_does_not_require_a_transcript(fake_gemini):
    response, generate, _ = fake_gemini
    response["transcript"] = ""
    result = service.extract_speech_features(b"webm", "audio/webm", "sustained_vowel")
    assert result.quality_status == "usable"
    assert "Selected recording task: sustained_vowel" in generate.call_args.kwargs["contents"][0]


def test_unusable_audio_is_not_turned_into_a_prediction_and_explains_why(fake_gemini):
    response, _, _ = fake_gemini
    response["quality_status"] = "unusable"
    response["quality_notes"] = ["Only background noise is audible."]
    with pytest.raises(AudioFeatureExtractionError, match="Only background noise is audible"):
        service.extract_speech_features(b"webm", "audio/webm")


def test_unusable_audio_without_notes_still_explains_recovery(fake_gemini):
    response, _, _ = fake_gemini
    response["quality_status"] = "unusable"
    with pytest.raises(AudioFeatureExtractionError, match="Listen to the preview"):
        service.extract_speech_features(b"webm", "audio/webm")
