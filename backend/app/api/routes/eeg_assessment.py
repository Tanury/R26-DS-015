"""EEG neurological risk endpoints.

Tier 1 (`/cohort`, `/model-card`, `/embeddings`) serves precomputed reports and
needs no ML runtime. Tier 2 (`/assessments`) runs the real pipeline and returns a
job id, because preprocessing takes 30-90 s and would exceed most proxy timeouts.
"""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.core.exceptions import (
    EegBundleError,
    EegIngestError,
    EegJobNotFoundError,
    EegSubjectNotFoundError,
)
from app.schemas.eeg_assessment import (
    BandReference,
    CohortPage,
    CohortProjection,
    EegJob,
    EegRiskReport,
    EmbeddingVector,
    ModelCard,
)
from app.services import eeg_cohort_service as cohort
from app.services import eeg_job_service as jobs


logger = logging.getLogger(__name__)
router = APIRouter()

RiskClass = Literal["AD", "PD", "MS", "HC"]
Site = Literal["AR", "CL"]
Quality = Literal["Good", "Moderate", "Poor"]


def _bundle_unavailable(exc: Exception) -> HTTPException:
    logger.exception("eeg_bundle_unavailable")
    return HTTPException(
        status_code=503,
        detail="The EEG model bundle is unavailable. Run scripts/build_eeg_cohort_index.py.",
    )


@router.get("/model-card", response_model=ModelCard, summary="EEG model card")
def model_card() -> ModelCard:
    try:
        return cohort.load_model_card()
    except EegBundleError as exc:
        raise _bundle_unavailable(exc) from exc


@router.get(
    "/band-reference",
    response_model=BandReference,
    summary="Cohort band-power statistics for contextualising one subject",
)
def band_reference() -> BandReference:
    """Per-class band medians and their separation from controls.

    Descriptive context, not attribution: the encoder never sees band power. A
    condition whose `has_signature` is False has no band-power pattern in this
    cohort and must not be given one.
    """
    try:
        return cohort.get_band_reference()
    except EegBundleError as exc:
        raise _bundle_unavailable(exc) from exc


@router.get("/cohort", response_model=CohortPage, summary="Browse the assessed cohort")
def cohort_index(
    search: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    true_class: Annotated[RiskClass | None, Query()] = None,
    site: Annotated[Site | None, Query()] = None,
    quality: Annotated[Quality | None, Query()] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> CohortPage:
    try:
        return cohort.list_cohort(
            search=search, true_class=true_class, site=site, quality=quality,
            offset=offset, limit=limit,
        )
    except EegBundleError as exc:
        raise _bundle_unavailable(exc) from exc


@router.get(
    "/cohort/projection",
    response_model=CohortProjection,
    summary="2-D projection of subject-level z_eeg",
)
def cohort_projection() -> CohortProjection:
    try:
        return cohort.get_projection()
    except EegBundleError as exc:
        raise _bundle_unavailable(exc) from exc


@router.get(
    "/cohort/{subject_id}",
    response_model=EegRiskReport,
    summary="Full risk report for one cohort subject",
)
def cohort_report(subject_id: str) -> EegRiskReport:
    try:
        return cohort.get_report(subject_id)
    except EegSubjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EegBundleError as exc:
        raise _bundle_unavailable(exc) from exc


@router.get(
    "/embeddings/{subject_id}",
    response_model=EmbeddingVector,
    summary="Full 256-D z_eeg vector for cross-modal fusion",
)
def embedding(subject_id: str) -> EmbeddingVector:
    try:
        return cohort.get_embedding(subject_id)
    except EegSubjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EegBundleError as exc:
        raise _bundle_unavailable(exc) from exc


@router.post(
    "/assessments",
    response_model=EegJob,
    status_code=202,
    summary="Upload an EEG recording for assessment",
)
async def create_assessment(
    files: Annotated[list[UploadFile], File(...)],
) -> EegJob:
    """Accepts an EEGLAB `.set`, optionally with its `.fdt` companion.

    Returns 202 with a job id immediately; poll `/assessments/{job_id}`.
    """
    if not cohort.load_model_card().inference_available:
        raise HTTPException(
            status_code=503,
            detail=(
                "EEG inference is not available on this deployment (PyTorch or the "
                "TorchScript graph is missing). Cohort browsing is unaffected."
            ),
        )

    try:
        payloads = await jobs.read_uploads(files, settings.max_eeg_bytes)
    except EegIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:  # oversize
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        return jobs.create_job(payloads)
    except RuntimeError as exc:  # queue saturated
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get(
    "/assessments/{job_id}",
    response_model=EegJob,
    summary="Poll an assessment job",
)
def assessment_status(job_id: str) -> EegJob:
    try:
        return jobs.get_job(job_id)
    except EegJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
