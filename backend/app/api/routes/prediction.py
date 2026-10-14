import logging

from fastapi import APIRouter, HTTPException

from app.core.exceptions import ModelLoadError, PredictionError
from app.schemas.biomedical_prediction_request import BiomedicalPredictionRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.biomedical_prediction_service import predict_biomedical_risk

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PredictionResponse)
def predict(payload: BiomedicalPredictionRequest) -> PredictionResponse:
    try:
        return predict_biomedical_risk(payload)
    except ModelLoadError as exc:
        logger.exception("prediction_model_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Prediction model is temporarily unavailable.",
        ) from exc
    except PredictionError as exc:
        logger.exception("prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from exc
