from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test audio-file dementia prediction either through the API or locally."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to a real audio file such as .wav, .mp3, .m4a, or .flac",
    )
    parser.add_argument(
        "--mode",
        choices=("api", "local"),
        default="api",
        help="Use 'api' to upload to the FastAPI endpoint or 'local' to run the model pipeline directly.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/v1/predict/audio-file",
        help="FastAPI audio prediction endpoint.",
    )
    return parser.parse_args()


def pretty_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_api_mode(audio_path: Path, url: str) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with audio_path.open("rb") as audio_file:
        files = {"file": (audio_path.name, audio_file, "application/octet-stream")}
        response = requests.post(url, files=files, timeout=300)

    print(f"HTTP {response.status_code}")
    try:
        pretty_print(response.json())
    except ValueError:
        print(response.text)


def run_local_mode(audio_path: Path) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    from app.services.audio_features import extract_audio_features
    from app.services.audio_predictor import predict_audio_features

    features = extract_audio_features(str(audio_path))
    result = predict_audio_features(features=features, file_id=audio_path.name)
    pretty_print(result)


def main() -> None:
    args = parse_args()

    if args.mode == "api":
        run_api_mode(args.audio_path, args.url)
    else:
        run_local_mode(args.audio_path)


if __name__ == "__main__":
    main()