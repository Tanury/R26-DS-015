from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RecordingTask = Literal[
    "diadochokinetic_task",
    "monologue",
    "picture_description",
    "reading",
    "sustained_vowel",
]


class NeurologicalRiskRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    rhythm_irregularity_score: float = Field(..., ge=0.0, le=1.0)
    voice_tremor_score: float = Field(..., ge=0.0, le=1.0)
    monotone_score: float = Field(..., ge=0.0, le=1.0)
    word_finding_pause_score: float = Field(..., ge=0.0, le=1.0)
    formant_instability_score: float = Field(..., ge=0.0, le=1.0)
    shimmer_local_pct: float = Field(..., ge=0.0)
    age: int = Field(..., ge=0, le=120)
    lexical_fluency_score: float = Field(..., ge=0.0, le=1.0)
    jitter_local_pct: float = Field(..., ge=0.0)
    mean_pause_duration_sec: float = Field(..., ge=0.0)
    disfluency_rate: float = Field(..., ge=0.0, le=1.0)
    total_pause_duration_sec: float = Field(..., ge=0.0)
    pause_count: int = Field(..., ge=0)
    pause_ratio: float = Field(..., ge=0.0, le=1.0)
    intensity_std_db: float = Field(..., ge=0.0)
    pitch_range_hz: float = Field(..., ge=0.0)
    articulation_rate_sps: float = Field(..., ge=0.0)
    pitch_std_hz: float = Field(..., ge=0.0)
    recording_task: RecordingTask
    mfcc1_mean: float
