import logging

from fastapi import APIRouter, HTTPException

from app.core.exceptions import ModelLoadError, PredictionError
from app.schemas.neurological_risk_request import NeurologicalRiskRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.neurological_prediction_service import (
    predict_neurological_risk,
)


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict neurological disease pattern and biomedical risk",
)
def predict(
    payload: NeurologicalRiskRequest,
) -> PredictionResponse:
    try:
        return predict_neurological_risk(payload)
    except ModelLoadError as exc:
        logger.exception("Neurological prediction model is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Neurological prediction model is temporarily unavailable.",
        ) from exc
    except PredictionError as exc:
        logger.exception("Neurological prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Unable to complete neurological prediction.",
        ) from exc
