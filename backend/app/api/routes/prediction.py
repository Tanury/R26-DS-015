import logging

from fastapi import APIRouter, HTTPException

from app.core.exceptions import FeatureValidationError, ModelLoadError, PredictionError
from app.schemas.prediction_request import PredictionRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.prediction_service import predict_risk

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        return predict_risk(payload)
    except FeatureValidationError as exc:
        logger.warning("prediction_rejected reason=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
