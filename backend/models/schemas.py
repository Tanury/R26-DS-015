"""
backend/models/schemas.py

Pydantic request/response models for the Vision Encoder API.
"""

from pydantic import BaseModel
from typing import Optional


class ClassConfidence(BaseModel):
    label: str         # "AD", "MCI", or "CN"
    confidence: float  # 0.0 - 1.0
    severity: int      # 0, 1, or 2


class MRIAnalysisResponse(BaseModel):
    # Classification result
    prediction: str              # "AD", "MCI", or "CN"
    prediction_full: str         # Human-readable label
    confidence: float            # Top class confidence 0-100
    risk_level: str              # "HIGH", "MODERATE", "LOW"

    # Per-class breakdown
    class_scores: list[ClassConfidence]

    # Embedding info
    embedding_dim: int           # Always 256
    embedding_norm: float        # Should be ~1.0 (L2 normalised)

    # Metadata
    filename: str
    modality: str                # "Brain MRI"
    model_version: str
    processing_time_ms: float
    disclaimer: str


class RetinalAnalysisResponse(BaseModel):
    prediction: str
    prediction_full: str
    confidence: float
    risk_level: str
    class_scores: list[ClassConfidence]
    embedding_dim: int
    filename: str
    modality: str
    model_version: str
    processing_time_ms: float
    disclaimer: str


DISCLAIMER = (
    "⚠️  This output is a computational feature representation for "
    "neurological risk analysis only. It is NOT a medical diagnosis. "
    "Always consult a qualified neurologist for clinical decisions."
)

PREDICTION_LABELS = {
    "AD":  "Alzheimer's Disease — significant neurodegeneration detected",
    "MCI": "Mild Cognitive Impairment — early-stage changes detected",
    "CN":  "Cognitively Normal — no significant abnormalities detected",
    "PD":  "Parkinson's Disease — dopaminergic changes detected",
    "HC":  "Healthy Control — no significant abnormalities detected",
    "MS":  "Multiple Sclerosis — white matter lesions detected",
}

RISK_MAP = {
    "AD": "HIGH",
    "MCI": "MODERATE",
    "CN": "LOW",
    "PD": "HIGH",
    "HC": "LOW",
    "MS": "HIGH",
}