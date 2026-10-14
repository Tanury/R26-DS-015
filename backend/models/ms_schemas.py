"""
backend/models/ms_schemas.py

Pydantic response schemas for Multiple Sclerosis analysis endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel


class ClassScore(BaseModel):
    label: str
    confidence: float


class ThicknessLayer(BaseModel):
    layer: str
    value_um: float
    hc_mean: Optional[float] = None
    ms_mean: Optional[float] = None

#new - ms mri
class RegionVolume(BaseModel):
    region: str
    value_frac: float     #fraction of total brain volume
    hc_mean: Optional[float] = None
    ms_mean: Optional[float] = None


class MSAnalysisResponse(BaseModel):
    prediction: str                              # "MS" or "HC"
    confidence: float                            # 0-100
    risk_level: str                               # "HIGH" | "MODERATE" | "LOW"
    class_scores: List[ClassScore]
    modality: str
    thickness: Optional[List[ThicknessLayer]] = None  #  for the OCT branch only
    regions: Optional[List[RegionVolume]] = None        #  for the MRI branch only
    slice_image: Optional[str] = None   #new, remove if not working !!!


class MSModelStatusResponse(BaseModel):
    oct_mode: str
    mri_mode: str
    oct_loaded: bool
    mri_loaded: bool
    message: str