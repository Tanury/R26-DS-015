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
