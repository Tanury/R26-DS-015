import io
import shutil
import subprocess
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.exceptions import AudioFeatureExtractionError
from app.services import audio_preparation as preparation


def pcm(seconds=3, amplitude=8000):
    samples = np.arange(int(seconds * preparation.SAMPLE_RATE))
    return (amplitude * np.sin(2 * np.pi * 220 * samples / preparation.SAMPLE_RATE)).astype("<i2").tobytes()


def stub_decoder(monkeypatch, output):
    monkeypatch.setattr(preparation.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(preparation.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))


def test_decodes_real_webm_opus_to_complete_wav():
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("FFmpeg is not installed")
    encoded = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-f", "s16le",
         "-ar", "16000", "-ac", "1", "-i", "pipe:0", "-c:a", "libopus", "-f", "webm", "pipe:1"],
        input=pcm(), capture_output=True, check=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    result, mime = preparation.prepare_browser_audio(encoded, "audio/webm")
    assert mime == "audio/wav"
    with wave.open(io.BytesIO(result), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16000
        assert abs(audio.getnframes() / 16000 - 3) < 0.1
        assert np.max(np.abs(np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2"))) > 1000


@pytest.mark.parametrize("mime", ["audio/webm", "audio/mp4", "audio/ogg"])
def test_browser_containers_are_sent_as_wav(monkeypatch, mime):
    stub_decoder(monkeypatch, pcm())
    result, result_mime = preparation.prepare_browser_audio(b"encoded-audio", mime)
    assert result[:4] == b"RIFF"
    assert result[8:12] == b"WAVE"
    assert result_mime == "audio/wav"


@pytest.mark.parametrize("seconds,amplitude,message", [(0, 8000, "too short"), (1, 8000, "too short"), (3, 0, "silent or too quiet"), (301, 8000, "five minutes")], ids=["empty", "short", "silent", "long"])
def test_rejects_unusable_recordings_before_remote_analysis(monkeypatch, seconds, amplitude, message):
    stub_decoder(monkeypatch, pcm(seconds, amplitude))
    with pytest.raises(AudioFeatureExtractionError, match=message):
        preparation.prepare_browser_audio(b"recording", "audio/webm")


def test_missing_decoder_gives_actionable_server_error(monkeypatch):
    monkeypatch.setattr(preparation.shutil, "which", lambda _: None)
    with pytest.raises(AudioFeatureExtractionError, match="not configured"):
        preparation.prepare_browser_audio(b"recording", "audio/webm")


@pytest.mark.parametrize("error,message", [
    (subprocess.TimeoutExpired("ffmpeg", 30), "too long to decode"),
    (subprocess.CalledProcessError(1, "ffmpeg"), "could not be decoded"),
])
def test_decoder_errors_are_actionable(monkeypatch, error, message):
    monkeypatch.setattr(preparation.shutil, "which", lambda _: "ffmpeg")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(preparation.subprocess, "run", fail)
    with pytest.raises(AudioFeatureExtractionError, match=message):
        preparation.prepare_browser_audio(b"broken", "audio/webm")


def test_other_audio_formats_are_preserved():
    assert preparation.prepare_browser_audio(b"original", "audio/wav") == (b"original", "audio/wav")
