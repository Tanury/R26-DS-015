"""
backend/routers/imaging_cohort.py

Serves the precomputed neuroimaging cohort (vision-encoder results on the
module's processed features) built by backend/scripts/build_imaging_cohort.py.

Endpoints:
    GET /api/imaging/cohort  — every cohort subject's prediction and z_img embedding
"""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

COHORT_PATH = Path(__file__).resolve().parents[1] / "app" / "models" / "imaging" / "cohort.json"


@lru_cache(maxsize=1)
def _load_cohort() -> dict:
    return json.loads(COHORT_PATH.read_text(encoding="utf-8"))


@router.get("/imaging/cohort", summary="Precomputed neuroimaging cohort results")
def imaging_cohort() -> dict:
    if not COHORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Imaging cohort not built. Run backend/scripts/build_imaging_cohort.py.",
        )
    return _load_cohort()
