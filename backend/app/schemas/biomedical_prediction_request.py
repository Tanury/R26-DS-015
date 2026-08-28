from typing import Literal

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:
    ConfigDict = None


EXAMPLE_PAYLOAD = {
    "sex": "Female",
    "apoe_genotype": "3/3",
    "alpha_synuclein_rt_quic_result": "Negative",
    "amyloid_beta_42_pg_ml": 850.0,
    "t_tau_pg_ml": 250.0,
    "p_tau181_pg_ml": 35.0,
    "nfl_pg_ml": 650.0,
    "gfap_pg_ml": 180.0,
    "alpha_synuclein_pg_ml": 1200.0,
    "kynurenic_acid_nm": 42.0,
    "quinolinic_acid_nm": 310.0,
    "gdf15_pg_ml": 760.0,
    "crp40_copy_number": 2.0,
    "dopamine_mrm_intensity": 12500.0,
}


class BiomedicalPredictionRequest(BaseModel):
    """Public input contract for the General Biomedical Risk Assessment."""

    sex: Literal["Female", "Male"]
    apoe_genotype: Literal["2/2", "2/3", "2/4", "3/3", "3/4", "4/4"]
    alpha_synuclein_rt_quic_result: Literal["Negative", "Positive"]

    amyloid_beta_42_pg_ml: float = Field(..., ge=0)
    t_tau_pg_ml: float = Field(..., ge=0)
    p_tau181_pg_ml: float = Field(..., ge=0)
    nfl_pg_ml: float = Field(..., ge=0)
    gfap_pg_ml: float = Field(..., ge=0)
    alpha_synuclein_pg_ml: float = Field(..., ge=0)
    kynurenic_acid_nm: float = Field(..., ge=0)
    quinolinic_acid_nm: float = Field(..., ge=0)
    gdf15_pg_ml: float = Field(..., ge=0)
    crp40_copy_number: float = Field(..., ge=0)
    dopamine_mrm_intensity: float = Field(..., ge=0)

    if hasattr(BaseModel, "model_validate") and ConfigDict is not None:
        model_config = ConfigDict(
            extra="forbid",
            allow_inf_nan=False,
            json_schema_extra={"example": EXAMPLE_PAYLOAD},
        )
    else:

        class Config:
            extra = "forbid"
            allow_inf_nan = False
            schema_extra = {"example": EXAMPLE_PAYLOAD}
