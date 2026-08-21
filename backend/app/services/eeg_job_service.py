"""In-process job registry for EEG assessments.

EEG preprocessing costs 30-90 s, so the upload endpoint returns 202 and the client
polls. A thread pool with a small bound keeps a burst of uploads from starving the
speech endpoints that share this process.

Deliberately in-memory: jobs are ephemeral by design and expire on a TTL. Nothing
here persists an uploaded recording beyond the job, which keeps the data-governance
question of storing patient EEG out of scope.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import (
    EegBundleError,
    EegIngestError,
    EegJobNotFoundError,
    EegQualityError,
)
from app.schemas.eeg_assessment import EegJob, JobError


_ALLOWED_SUFFIXES = {".set", ".fdt"}
_SET_MAGIC = b"MATLAB"          # EEGLAB .set files are MATLAB v5/v7 containers
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"  # MATLAB v7.3 is HDF5

_jobs: dict[str, EegJob] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(
    max_workers=max(1, settings.eeg_max_active_jobs), thread_name_prefix="eeg-job"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def read_uploads(files: list[UploadFile], max_bytes: int) -> dict[str, bytes]:
    """Validate and buffer the uploaded EEGLAB pair.

    Raises ValueError for oversize (413) and EegIngestError for anything else (422).
    """
    if not files:
        raise EegIngestError("No files were uploaded.")
    if len(files) > 2:
        raise EegIngestError("Upload at most two files: one .set and its .fdt companion.")

    payloads: dict[str, bytes] = {}
    total = 0
    for upload in files:
        name = Path(upload.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            await upload.close()
            raise EegIngestError(
                f"Unsupported file '{name}'. Upload an EEGLAB .set file "
                "and, when the data is stored separately, its .fdt companion."
            )
        data = await upload.read(max_bytes + 1)
        await upload.close()
        total += len(data)
        if not data:
            raise EegIngestError(f"'{name}' is empty.")
        if total > max_bytes:
            raise ValueError(
                f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit."
            )
        payloads[name] = data

    set_files = [n for n in payloads if n.lower().endswith(".set")]
    if len(set_files) != 1:
        raise EegIngestError("Exactly one .set file is required.")

    header = payloads[set_files[0]][:8]
    if not (header.startswith(_SET_MAGIC) or header.startswith(_HDF5_MAGIC)):
        raise EegIngestError(
            "The .set file does not look like an EEGLAB/MATLAB container."
        )

    # A .set without embedded data needs its .fdt; catching it here gives a far
    # better message than a FileNotFoundError deep inside MNE.
    set_name = set_files[0]
    has_fdt = any(n.lower().endswith(".fdt") for n in payloads)
    if not has_fdt and len(payloads[set_name]) < 10 * 1024 * 1024:
        raise EegIngestError(
            f"'{set_name}' is only {len(payloads[set_name]) // 1024} KB, which means it is "
            "a header that references a separate .fdt data file. Upload both files together."
        )
    return payloads


def create_job(payloads: dict[str, bytes]) -> EegJob:
    with _lock:
        _purge_expired()
        active = sum(
            1 for job in _jobs.values()
            if job.status in {"queued", "validating", "preprocessing", "inference"}
        )
        if active >= settings.eeg_max_active_jobs:
            raise RuntimeError(
                "Too many EEG assessments are already running. Try again shortly."
            )

        job_id = uuid.uuid4().hex
        set_name = next(n for n in payloads if n.lower().endswith(".set"))
        job = EegJob(
            job_id=job_id, status="queued", progress=0,
            stage_label="Queued", filename=set_name,
            created_at=_now(), updated_at=_now(),
        )
        _jobs[job_id] = job

    _executor.submit(_run_job, job_id, payloads)
    return job


def get_job(job_id: str) -> EegJob:
    with _lock:
        _purge_expired()
        job = _jobs.get(job_id)
    if job is None:
        raise EegJobNotFoundError(f"Unknown or expired assessment job '{job_id}'.")
    return job


def _update(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        _jobs[job_id] = job.model_copy(update={**fields, "updated_at": _now()})


def _purge_expired() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.eeg_job_ttl_seconds)
    for job_id, job in list(_jobs.items()):
        try:
            updated = datetime.fromisoformat(job.updated_at)
        except ValueError:
            continue
        if updated < cutoff:
            _jobs.pop(job_id, None)


def _run_job(job_id: str, payloads: dict[str, bytes]) -> None:
    # Imported here so a deployment without PyTorch/MNE can still serve Tier 1.
    from app.services.eeg_inference_service import infer
    from app.services.eeg_model_loader import load_eeg_assets
    from app.services.eeg_preprocessing import preprocess

    workdir = Path(tempfile.mkdtemp(prefix=f"eeg-{job_id}-"))
    try:
        _update(job_id, status="validating", progress=2, stage_label="Validating upload")
        for name, data in payloads.items():
            (workdir / Path(name).name).write_bytes(data)
        set_path = next(workdir.glob("*.set"))

        assets = load_eeg_assets()
        config = {
            **assets.preprocessing_config,
            "channel_order": list(assets.channel_order),
            "sampling_rate_hz": assets.sampling_rate_hz,
            "epoch_length_seconds": assets.epoch_length_seconds,
            "standardization": assets.standardization,
        }

        def progress(pct: int, label: str) -> None:
            _update(job_id, status="preprocessing", progress=pct, stage_label=label)

        _update(job_id, status="preprocessing", progress=5, stage_label="Reading recording")
        preprocessed = preprocess(set_path, config, progress=progress)

        _update(job_id, status="inference", progress=90, stage_label="Running the encoder")
        report = infer(preprocessed, subject_id=set_path.stem, explain=True)

        _update(job_id, status="completed", progress=100,
                stage_label="Complete", report=report)

    except EegQualityError as exc:
        _update(job_id, status="failed", progress=100, stage_label="Rejected",
                error=JobError(code="insufficient_quality", message=str(exc),
                               details=getattr(exc, "diagnostics", {}) or {}))
    except EegIngestError as exc:
        _update(job_id, status="failed", progress=100, stage_label="Rejected",
                error=JobError(code="unreadable_recording", message=str(exc)))
    except EegBundleError as exc:
        _update(job_id, status="failed", progress=100, stage_label="Unavailable",
                error=JobError(code="model_unavailable", message=str(exc)))
    except Exception as exc:  # noqa: BLE001 - job boundary; never leak a traceback
        _update(job_id, status="failed", progress=100, stage_label="Failed",
                error=JobError(code="inference_failed",
                               message=f"{type(exc).__name__}: {exc}"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def reset_jobs() -> None:
    """Test helper — clears the registry."""
    with _lock:
        _jobs.clear()
