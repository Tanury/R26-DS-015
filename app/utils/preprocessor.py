"""
utils/preprocessor.py
──────────────────────
Converts a raw PatientInput → feature matrix ready for model inference.

Steps:
  1. Convert Pydantic model to dict
  2. Auto-fill missingness indicator flags from None values
  3. Label-encode categorical columns using trained encoders
  4. Impute remaining numeric NaNs using trained median imputer
  5. Return a single-row DataFrame in the exact column order used during training
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

from app.config import CATEGORICAL_COLS, NUMERIC_COLS, ALL_FEATURE_COLS
from app.schemas import PatientInput


# Mapping: raw biomarker → its _missing indicator column name
MISSINGNESS_PAIRS: Dict[str, str] = {
    "amyloid_beta_42_pg_ml":          "amyloid_beta_42_pg_ml_missing",
    "amyloid_beta_42_40_ratio":       "amyloid_beta_42_40_ratio_missing",
    "t_tau_pg_ml":                    "t_tau_pg_ml_missing",
    "p_tau181_pg_ml":                 "p_tau181_pg_ml_missing",
    "nfl_pg_ml":                      "nfl_pg_ml_missing",
    "gfap_pg_ml":                     "gfap_pg_ml_missing",
    "hemoglobin_ng_ml":               "hemoglobin_ng_ml_missing",
    "gdf15_pg_ml":                    "gdf15_pg_ml_missing",
    "klotho_pg_ml":                   "klotho_pg_ml_missing",
    "crp40_copy_number":              "crp40_copy_number_missing",
    "alpha_synuclein_seed_auc":       "alpha_synuclein_seed_auc_missing",
    "alpha_synuclein_t50_hours":      "alpha_synuclein_t50_hours_missing",
    "alpha_synuclein_rt_quic_result": "alpha_synuclein_rt_quic_result_missing",
    "ceramide_total_nmol_ml":         "ceramide_total_nmol_ml_missing",
    "glucosylceramide_total_nmol_ml": "glucosylceramide_total_nmol_ml_missing",
    "lactosylceramide_total_nmol_ml": "lactosylceramide_total_nmol_ml_missing",
    "sphingomyelin_total_nmol_ml":    "sphingomyelin_total_nmol_ml_missing",
    "tryptophan_nm":                  "tryptophan_nm_missing",
    "kynurenine_nm":                  "kynurenine_nm_missing",
    "kynurenic_acid_nm":              "kynurenic_acid_nm_missing",
    "quinolinic_acid_nm":             "quinolinic_acid_nm_missing",
    "serotonin_nm":                   "serotonin_nm_missing",
    "dopamine_mrm_intensity":         "dopamine_mrm_intensity_missing",
    "creatine_mrm_intensity":         "creatine_mrm_intensity_missing",
    "olink_mdk_npx":                  "olink_mdk_npx_missing",
    "olink_il17d_npx":                "olink_il17d_npx_missing",
    "olink_ddc_npx":                  "olink_ddc_npx_missing",
    "olink_mad5_npx":                 "olink_mad5_npx_missing",
    "mir_107_ct":                     "mir_107_ct_missing",
    "mir_132_ct":                     "mir_132_ct_missing",
    "mir_146a_ct":                    "mir_146a_ct_missing",
    "mir_155_ct":                     "mir_155_ct_missing",
    "mir_9_ct":                       "mir_9_ct_missing",
    "mir_let7e_ct":                   "mir_let7e_ct_missing",
    "apoe_genotype":                  "apoe_genotype_missing",
    "gba_mutation_carrier":           "gba_mutation_carrier_missing",
    "gcase_activity":                 "gcase_activity_missing",
    "alpha_synuclein_pg_ml":          "alpha_synuclein_pg_ml_missing",
    "mir_107_relative_expression":    "mir_107_relative_expression_missing",
    "mir_132_relative_expression":    "mir_132_relative_expression_missing",
    "mir_146a_relative_expression":   "mir_146a_relative_expression_missing",
    "mir_155_relative_expression":    "mir_155_relative_expression_missing",
    "mir_9_relative_expression":      "mir_9_relative_expression_missing",
    "mir_let7e_relative_expression":  "mir_let7e_relative_expression_missing",
}


def preprocess(
    patient: PatientInput,
    cat_encoders: Dict,
    num_imputer,
) -> pd.DataFrame:
    """
    Transform a PatientInput into a model-ready single-row DataFrame.

    Parameters
    ----------
    patient     : PatientInput  — validated Pydantic model
    cat_encoders: dict          — {col: sklearn LabelEncoder} from training
    num_imputer : SimpleImputer — fitted median imputer from training

    Returns
    -------
    pd.DataFrame with shape (1, n_features), columns in training order.
    """

    # 1. Pydantic → plain dict
    data: Dict[str, Any] = patient.model_dump()

    # 2. Auto-fill missingness indicator flags
    #    If caller didn't provide the _missing flag, derive it from the raw value
    for raw_col, miss_col in MISSINGNESS_PAIRS.items():
        if miss_col in data and data[miss_col] is None:
            data[miss_col] = 1 if data.get(raw_col) is None else 0

    # 3. Build DataFrame — only columns the model knows about
    row = pd.DataFrame([data])

    # 4. Encode categorical columns
    for col in CATEGORICAL_COLS:
        if col not in row.columns or row[col].isna().all():
            row[col] = 0
            continue

        enc = cat_encoders.get(col)
        if enc is None:
            row[col] = 0
            continue

        raw_val = str(row[col].fillna("Missing").values[0])
        if raw_val in enc.classes_:
            row[col] = enc.transform([raw_val])[0]
        else:
            # Unseen category → use 0 (most common during training)
            row[col] = 0

    # 5. Ensure all expected columns are present (add as NaN if missing)
    for col in ALL_FEATURE_COLS:
        if col not in row.columns:
            row[col] = np.nan

    # 6. Reorder to exact training column order
    row = row[ALL_FEATURE_COLS]

    # 7. Impute remaining numeric NaNs
    row[NUMERIC_COLS] = num_imputer.transform(row[NUMERIC_COLS])

    return row
