"""Tier 1 — serve precomputed per-subject EEG reports.

This module deliberately imports nothing heavier than the standard library plus
pydantic. Browsing the cohort must work on a deployment that has no PyTorch and no
MNE installed, because it reads JSON the training notebook already wrote.

The confound disclosure is re-merged from `model_card.json` on every read rather
than trusted from the stored report. Those numbers describe the *model*, not the
subject: if the encoder is retrained and the age probe improves, stored reports
would otherwise keep quoting stale figures forever.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import EegBundleError, EegSubjectNotFoundError
from app.schemas.eeg_assessment import (
    BandReference,
    CohortPage,
    CohortProjection,
    CohortSubject,
    ConfoundDisclosure,
    EegRiskReport,
    EmbeddingVector,
    ModelCard,
)


INDEX_FILENAME = "index.json"
PROJECTION_FILENAME = "projection.json"
BAND_REFERENCE_FILENAME = "band_reference.json"
MODEL_CARD_FILENAME = "model_card.json"
CRITICAL_MARKER = "CRITICAL"


def _cohort_dir() -> Path:
    return Path(settings.eeg_model_dir) / "cohort"


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise EegBundleError(f"Missing EEG bundle file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise EegBundleError(f"EEG bundle file is not valid JSON: {path.name}") from exc


@lru_cache(maxsize=1)
def load_model_card() -> ModelCard:
    """Model card plus a runtime flag for whether inference is actually available."""
    payload = _read_json(Path(settings.eeg_model_dir) / MODEL_CARD_FILENAME)
    disclosure = _build_disclosure(payload.get("confound_disclosure", {}))

    model = payload.get("model", {})
    try:
        return ModelCard(
            run_id=str(payload.get("run_id", "unknown")),
            generated_at=payload.get("generated_at"),
            architecture=str(model.get("architecture", "unknown")),
            input_representation=str(model.get("input_representation", "unknown")),
            input_shape=[int(v) for v in model.get("input_shape", [])],
            embedding_dim=int(model.get("embedding_dim", 256)),
            risk_conditions=[
                str(c) for c in model.get("outputs", {})
                .get("risk_scores", {})
                .get("order", ["AD", "PD", "MS"])
            ],
            cohort=payload.get("training_data", {}),
            performance=payload.get("performance", {}),
            confound_disclosure=disclosure,
            intended_use=payload.get("intended_use", {}),
            inference_available=_inference_available(),
        )
    except Exception as exc:  # malformed card
        raise EegBundleError(f"EEG model card is malformed: {exc}") from exc


def _inference_available() -> bool:
    """Tier 2 needs PyTorch and the TorchScript graph; Tier 1 needs neither."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return (Path(settings.eeg_model_dir) / "neuro_risk_encoder.torchscript.pt").exists()


def _build_disclosure(raw: dict[str, Any]) -> ConfoundDisclosure:
    severity = {
        str(k): str(v) for k, v in (raw.get("severity_by_condition") or {}).items()
    }
    if not severity:
        severity = {
            str(k): str(v) for k, v in (raw.get("per_class_severity") or {}).items()
        }

    age_probe = raw.get("age_probe") or {}
    site_probe = raw.get("site_probe") or {}
    correlations = {
        str(condition): float(entry.get("pearson_r_within_negatives",
                                        entry.get("pearson_r_score_vs_age", 0.0)))
        for condition, entry in (raw.get("risk_score_age_correlation") or {}).items()
        if isinstance(entry, dict)
    }

    return ConfoundDisclosure(
        age_probe_mae_years=_maybe_float(
            raw.get("age_probe_mae_years", age_probe.get("mae_years"))),
        age_probe_improvement_over_baseline=_maybe_float(
            raw.get("age_probe_improvement_over_baseline",
                    age_probe.get("improvement_over_baseline"))),
        site_probe_balanced_accuracy=_maybe_float(
            raw.get("site_probe_balanced_accuracy",
                    site_probe.get("balanced_accuracy"))),
        risk_score_age_correlation=correlations,
        severity_by_condition=severity,
        statement=str(raw.get("statement", "")),
        has_critical=any(CRITICAL_MARKER in value for value in severity.values()),
    )


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def load_cohort_index() -> list[CohortSubject]:
    rows = _read_json(_cohort_dir() / INDEX_FILENAME)
    if not isinstance(rows, list) or not rows:
        raise EegBundleError("EEG cohort index is empty or malformed.")
    try:
        return [CohortSubject(**row) for row in rows]
    except Exception as exc:
        raise EegBundleError(f"EEG cohort index row is malformed: {exc}") from exc


def list_cohort(
    *,
    true_class: str | None = None,
    site: str | None = None,
    quality: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> CohortPage:
    subjects = load_cohort_index()
    filtered = [
        subject for subject in subjects
        if (true_class is None or subject.true_class == true_class)
        and (site is None or subject.site == site)
        and (quality is None or subject.signal_quality == quality)
    ]
    window = filtered[offset: offset + limit]
    return CohortPage(
        total=len(filtered),
        offset=offset,
        limit=limit,
        subjects=window,
        available_filters={
            "true_class": sorted({s.true_class for s in subjects}),
            "site": sorted({s.site for s in subjects}),
            "signal_quality": sorted({s.signal_quality for s in subjects}),
        },
    )


def get_report(subject_id: str) -> EegRiskReport:
    """Full report for one subject, with the model-level disclosure re-merged."""
    safe_id = _safe_subject_id(subject_id)
    path = _cohort_dir() / f"{safe_id}.json"
    if not path.exists():
        raise EegSubjectNotFoundError(f"No EEG report for subject '{subject_id}'.")

    payload = _read_json(path)
    payload["confound_disclosure"] = load_model_card().confound_disclosure.model_dump()
    payload.setdefault("source", "cohort")
    payload.setdefault("clinical_disclaimer", settings.eeg_disclaimer)
    try:
        return EegRiskReport(**payload)
    except Exception as exc:
        raise EegBundleError(
            f"Stored report for '{subject_id}' does not match the schema: {exc}"
        ) from exc


def get_embedding(subject_id: str) -> EmbeddingVector:
    safe_id = _safe_subject_id(subject_id)
    path = _cohort_dir() / "embeddings" / f"{safe_id}.json"
    if not path.exists():
        raise EegSubjectNotFoundError(f"No z_eeg vector for subject '{subject_id}'.")
    payload = _read_json(path)
    try:
        return EmbeddingVector(**payload)
    except Exception as exc:
        raise EegBundleError(f"Stored z_eeg for '{subject_id}' is malformed: {exc}") from exc


@lru_cache(maxsize=1)
def get_projection() -> CohortProjection:
    payload = _read_json(_cohort_dir() / PROJECTION_FILENAME)
    try:
        return CohortProjection(**payload)
    except Exception as exc:
        raise EegBundleError(f"EEG cohort projection is malformed: {exc}") from exc


@lru_cache(maxsize=1)
def get_band_reference() -> BandReference:
    """Cohort band-power statistics used to contextualise a single subject.

    Separate from the model card because it describes the *recordings*, not the
    encoder — and because it is the one payload that can legitimately tell a reader
    "this condition has no band-power pattern here".
    """
    payload = _read_json(_cohort_dir() / BAND_REFERENCE_FILENAME)
    try:
        return BandReference(**payload)
    except Exception as exc:
        raise EegBundleError(f"EEG band reference is malformed: {exc}") from exc


def _safe_subject_id(subject_id: str) -> str:
    """Reject anything that could escape the cohort directory."""
    candidate = str(subject_id).strip()
    if not candidate or len(candidate) > 128:
        raise EegSubjectNotFoundError("Invalid subject id.")
    if not all(char.isalnum() or char in "-_" for char in candidate):
        raise EegSubjectNotFoundError("Invalid subject id.")
    return candidate


def reset_caches() -> None:
    """Drop cached bundle state — used by tests and after a bundle swap."""
    load_model_card.cache_clear()
    load_cohort_index.cache_clear()
    get_projection.cache_clear()
    get_band_reference.cache_clear()
