"""Upload validation and the job registry.

These run without PyTorch: every case here is rejected before inference is reached,
which is exactly the boundary worth testing hardest.
"""

import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import EegIngestError, EegJobNotFoundError
from app.main import app
from app.schemas.eeg_assessment import EegJob
from app.services import eeg_cohort_service as cohort
from app.services import eeg_job_service as jobs


client = TestClient(app)

SET_MAGIC = b"MATLAB 5.0 MAT-file, Platform: PCWIN64"
BIG_SET = SET_MAGIC + b"\x00" * (11 * 1024 * 1024)   # >10 MB: self-contained


@pytest.fixture(autouse=True)
def _reset():
    cohort.reset_caches()
    jobs.reset_jobs()
    yield
    jobs.reset_jobs()


class _FakeUpload:
    """Minimal stand-in for starlette's UploadFile."""

    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._buffer = io.BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    async def close(self) -> None:
        self._buffer.close()


def _read(files, max_bytes=120 * 1024 * 1024):
    return asyncio.run(jobs.read_uploads(files, max_bytes))


def test_accepts_self_contained_set() -> None:
    payloads = _read([_FakeUpload("sub-30001.set", BIG_SET)])
    assert "sub-30001.set" in payloads


def test_accepts_set_plus_fdt_pair() -> None:
    payloads = _read([
        _FakeUpload("sub-100012.set", SET_MAGIC + b"\x00" * 2048),
        _FakeUpload("sub-100012.fdt", b"\x00" * 4096),
    ])
    assert len(payloads) == 2


def test_rejects_header_only_set_without_fdt() -> None:
    """The real failure in this dataset: a 3 MB header whose 90 MB .fdt never arrived."""
    with pytest.raises(EegIngestError) as excinfo:
        _read([_FakeUpload("sub-100013.set", SET_MAGIC + b"\x00" * 2048)])
    assert ".fdt" in str(excinfo.value)


def test_rejects_wrong_extension() -> None:
    with pytest.raises(EegIngestError):
        _read([_FakeUpload("recording.edf", b"0" * 4096)])


def test_rejects_non_matlab_container() -> None:
    with pytest.raises(EegIngestError):
        _read([_FakeUpload("fake.set", b"NOT-A-MATLAB-FILE" + b"\x00" * (11 * 1024 * 1024))])


def test_rejects_empty_file() -> None:
    with pytest.raises(EegIngestError):
        _read([_FakeUpload("sub.set", b"")])


def test_rejects_oversize_upload() -> None:
    with pytest.raises(ValueError):
        _read([_FakeUpload("sub.set", b"M" * 4096)], max_bytes=1024)


def test_requires_exactly_one_set() -> None:
    with pytest.raises(EegIngestError):
        _read([_FakeUpload("a.fdt", b"\x00" * 512)])


def test_unknown_job_returns_404() -> None:
    assert client.get("/eeg/assessments/deadbeef").status_code == 404


def test_job_registry_roundtrip(monkeypatch) -> None:
    """create_job registers a queued job without needing the worker to succeed."""
    monkeypatch.setattr(jobs._executor, "submit", lambda *a, **k: None)
    job = jobs.create_job({"sub-x.set": BIG_SET})
    assert isinstance(job, EegJob)
    assert job.status == "queued"
    assert jobs.get_job(job.job_id).job_id == job.job_id


def test_job_queue_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(jobs._executor, "submit", lambda *a, **k: None)
    monkeypatch.setattr(jobs.settings, "eeg_max_active_jobs", 2)
    jobs.create_job({"a.set": BIG_SET})
    jobs.create_job({"b.set": BIG_SET})
    with pytest.raises(RuntimeError):
        jobs.create_job({"c.set": BIG_SET})


def test_expired_jobs_are_purged(monkeypatch) -> None:
    monkeypatch.setattr(jobs._executor, "submit", lambda *a, **k: None)
    job = jobs.create_job({"a.set": BIG_SET})
    monkeypatch.setattr(jobs.settings, "eeg_job_ttl_seconds", -1)
    with pytest.raises(EegJobNotFoundError):
        jobs.get_job(job.job_id)
