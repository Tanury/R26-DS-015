from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NeurologicalRiskRequest(BaseModel):
    """Exact 24-feature contract recorded by the neurological model metadata."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    # Every key is required. A null value is valid and is imputed by the fitted
    # pipeline using its training-set statistic.
    age: int | None = Field(..., ge=18, le=120)
    sex: Literal["Female", "Male"] | None = Field(...)
    education_years: float | None = Field(..., ge=0.0, le=40.0)
    bmi: float | None = Field(..., ge=10.0, le=80.0)
    family_history_pd: int | None = Field(..., ge=0, le=1)
    systolic_bp: float | None = Field(..., ge=50.0, le=300.0)
    diastolic_bp: float | None = Field(..., ge=30.0, le=200.0)

    cognitive_screen_score_0_30: float | None = Field(..., ge=0.0, le=30.0)
    rem_sleep_score: float | None = Field(..., ge=0.0, le=20.0)
    updrs_part_i: float | None = Field(..., ge=0.0, le=52.0)
    updrs_part_ii: float | None = Field(..., ge=0.0, le=52.0)
    updrs_part_iii: float | None = Field(..., ge=0.0, le=132.0)
    updrs_part_iv: float | None = Field(..., ge=0.0, le=24.0)
    schwab_england_adl: float | None = Field(..., ge=0.0, le=100.0)

    apoe_e4_count: int | None = Field(..., ge=0, le=2)
    gba_variant_carrier: int | None = Field(..., ge=0, le=1)

    amyloid_beta_42_40_ratio: float | None = Field(..., ge=0.0, le=1.0)
    t_tau_pg_ml: float | None = Field(..., ge=0.0)
    p_tau181_pg_ml: float | None = Field(..., ge=0.0)
    nfl_pg_ml: float | None = Field(..., ge=0.0)
    gfap_pg_ml: float | None = Field(..., ge=0.0)
    alpha_synuclein_pg_ml: float | None = Field(..., ge=0.0)
    gdf15_pg_ml: float | None = Field(..., ge=0.0)
    crp40_copy_number: float | None = Field(..., ge=0.0)
