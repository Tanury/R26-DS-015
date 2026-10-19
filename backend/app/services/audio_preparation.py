"""Decode browser recordings to a bounded, seekable PCM WAV for extraction."""

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from app.core.exceptions import AudioFeatureExtractionError


BROWSER_AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/mp4"}
SAMPLE_RATE = 16000
MAX_SECONDS = 300


def prepare_browser_audio(audio: bytes, mime_type: str) -> tuple[bytes, str]:
    if mime_type not in BROWSER_AUDIO_TYPES:
        return audio, mime_type

    executable = shutil.which("ffmpeg")
    if not executable:
        raise AudioFeatureExtractionError(
            "Browser audio decoding is not configured on the server. "
            "Install FFmpeg and add it to PATH, or upload a WAV recording."
        )

    # A local file also supports MP4 recordings whose metadata requires seeking.
    # The decoder cannot fetch external URLs, and never uses a shell command.
    with tempfile.TemporaryDirectory(prefix="voice-audio-") as directory:
        source = Path(directory) / "recording.bin"
        source.write_bytes(audio)
        try:
            decoded = subprocess.run(
                [
                    executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-protocol_whitelist", "file,pipe", "-i", str(source),
                    "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 1),
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
                ],
                capture_output=True,
                timeout=30,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioFeatureExtractionError(
                "The recording took too long to decode. Record a shorter sample and try again."
            ) from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            raise AudioFeatureExtractionError(
                "The recording could not be decoded. Listen to the preview, then record again or upload a WAV file."
            ) from exc

    pcm = decoded.stdout
    duration = len(pcm) / (SAMPLE_RATE * 2)
    if duration > MAX_SECONDS:
        raise AudioFeatureExtractionError("Recordings must be five minutes or shorter.")
    if duration < 2:
        raise AudioFeatureExtractionError(
            "The recording is too short. Record at least two seconds of voice; "
            "for the reading task, read the full passage."
        )
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    if np.max(np.abs(samples)) <= 2:
        raise AudioFeatureExtractionError(
            "The microphone recording is silent or too quiet. Check the microphone level, "
            "unmute the correct microphone, and record again."
        )

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue(), "audio/wav"
