from fastapi import APIRouter

from app.api.routes.eeg_assessment import router as eeg_router
from app.api.routes.health import router as health_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.voice_assessment import router as voice_assessment_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(prediction_router, prefix="/predictions", tags=["predictions"])
api_router.include_router(
    voice_assessment_router,
    prefix="/voice-assessments",
    tags=["voice-assessments"],
)
api_router.include_router(eeg_router, prefix="/eeg", tags=["eeg-risk"])
