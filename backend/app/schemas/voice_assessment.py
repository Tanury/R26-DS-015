from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prediction_response import PredictionResponse


class ExtractedSpeechFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mfcc_1_mean: float = Field(..., ge=-1000, le=1000)
    mfcc_2_mean: float = Field(..., ge=-1000, le=1000)
    mfcc_3_mean: float = Field(..., ge=-1000, le=1000)
    pitch_mean: float = Field(..., ge=20, le=1000)
    pitch_std: float = Field(..., ge=0, le=500)
    jitter: float = Field(..., ge=0, le=100)
    shimmer: float = Field(..., ge=0, le=100)
    hnr: float = Field(..., ge=-30, le=100)
    speech_rate: float = Field(..., ge=0, le=12)
    pause_count: float = Field(..., ge=0, le=1000)
    mean_pause_duration: float = Field(..., ge=0, le=60)
    mean_energy: float = Field(..., ge=-150, le=150)
    spectral_centroid_mean: float = Field(..., ge=0, le=30000)
    zero_crossing_rate_mean: float = Field(..., ge=0, le=1)


class GeminiAudioAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_status: Literal["usable", "limited", "unusable"]
    transcript: str = Field(default="", max_length=4000)
    quality_notes: list[str] = Field(default_factory=list, max_length=6)
    features: ExtractedSpeechFeatures


class VoiceAssessmentResponse(BaseModel):
    filename: str
    recording_task: str
    patient_age: int
    extraction_quality: Literal["usable", "limited"]
    quality_notes: list[str]
    transcript: str
    extracted_features: ExtractedSpeechFeatures
    prediction: PredictionResponse
    extraction_disclaimer: str
