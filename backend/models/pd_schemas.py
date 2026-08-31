"""
backend/models/pd_schemas.py

Pydantic response models for the Parkinson's Disease analysis endpoints.
Mirrors the structure of schemas.py used by the AD branch.
"""

from pydantic import BaseModel
from typing import Optional


class PDClassConfidence(BaseModel):
    label:      str    # "PD" or "HC"
    confidence: float  # 0.0 – 1.0
    severity:   int    # 1 = PD (high risk), 0 = HC (low risk)


class PDAnalysisResponse(BaseModel):
    # Classification result
    prediction:      str    # "PD" or "HC"
    prediction_full: str    # Human-readable label
    confidence:      float  # Top class confidence 0–100
    risk_level:      str    # "HIGH" or "LOW"

    # Per-class breakdown
    class_scores: list[PDClassConfidence]

    # Embedding info (for fusion engine)
    embedding_dim:  int    # Always 256
    embedding_norm: float  # Should be ~1.0 (L2 normalised)

    # Metadata
    filename:             str
    modality:             str    # "Brain MRI", "DaTscan SPECT", or "Multimodal"
    model_version:        str
    processing_time_ms:   float
    disclaimer:           str


class PDModelStatusResponse(BaseModel):
    mri_mode:       str   # "real" or "mock"
    dat_mode:       str   # "real" or "mock"
    mri_loaded:     bool
    dat_loaded:     bool
    message:        str
