"""
config.py
─────────
Central configuration for BIO_BACKEND.
All paths, constants, and environment settings live here.
"""

import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
MODELS_DIR = BASE_DIR / "app" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "saved_models"

# ── Saved model file paths ────────────────────────────────────────────────────
DISEASE_MODEL_PATH       = MODELS_DIR / "disease_model.pkl"
DISEASE_LABEL_ENC_PATH   = MODELS_DIR / "disease_label_encoder.pkl"
DISEASE_CAT_ENC_PATH     = MODELS_DIR / "disease_cat_encoders.pkl"
DISEASE_IMPUTER_PATH     = MODELS_DIR / "disease_num_imputer.pkl"
DISEASE_SHAP_PATH        = MODELS_DIR / "disease_shap_importance.csv"

RISK_MODEL_PATH          = MODELS_DIR / "risk_model.pkl"
RISK_LABEL_MAP_PATH      = MODELS_DIR / "risk_label_mapping.pkl"
RISK_CAT_ENC_PATH        = MODELS_DIR / "risk_cat_encoders.pkl"
RISK_IMPUTER_PATH        = MODELS_DIR / "risk_num_imputer.pkl"
RISK_SHAP_PATH           = MODELS_DIR / "risk_shap_importance.csv"

# Backward-compatible aliases for older import names.
DISEASE_LABEL_ENCODER_PATH = DISEASE_LABEL_ENC_PATH
DISEASE_CAT_ENCODERS_PATH = DISEASE_CAT_ENC_PATH
DISEASE_NUM_IMPUTER_PATH = DISEASE_IMPUTER_PATH
RISK_LABEL_MAPPING_PATH = RISK_LABEL_MAP_PATH
RISK_CAT_ENCODERS_PATH = RISK_CAT_ENC_PATH
RISK_NUM_IMPUTER_PATH = RISK_IMPUTER_PATH

# ── Feature columns (must match training order exactly) ───────────────────────
CATEGORICAL_COLS = [
    "sex",
    "sample_type",
    "apoe_genotype",
    "alpha_synuclein_rt_quic_result",
]

NUMERIC_COLS = [
    "age", "education_years", "family_history_pd", "disease_duration_years",
    "levodopa_use", "systolic_bp", "diastolic_bp", "bmi", "moca_total_score",
    "rem_sleep_score", "updrs_part_i", "updrs_part_ii", "updrs_part_iii",
    "updrs_part_iv", "schwab_england_adl", "gba_mutation_carrier",
    "amyloid_beta_42_pg_ml", "amyloid_beta_42_40_ratio", "t_tau_pg_ml",
    "p_tau181_pg_ml", "nfl_pg_ml", "gfap_pg_ml", "alpha_synuclein_pg_ml",
    "alpha_synuclein_seed_auc", "alpha_synuclein_t50_hours",
    "hemoglobin_ng_ml", "gdf15_pg_ml", "klotho_pg_ml", "crp40_copy_number",
    "gcase_activity", "beta_hexosaminidase_activity", "cathepsin_d_activity",
    "ceramide_total_nmol_ml", "glucosylceramide_total_nmol_ml",
    "lactosylceramide_total_nmol_ml", "sphingomyelin_total_nmol_ml",
    "tryptophan_nm", "kynurenine_nm", "kynurenic_acid_nm",
    "quinolinic_acid_nm", "serotonin_nm", "dopamine_mrm_intensity",
    "creatine_mrm_intensity", "olink_mdk_npx", "olink_il17d_npx",
    "olink_ddc_npx", "olink_mad5_npx", "mir_107_ct", "mir_132_ct",
    "mir_146a_ct", "mir_155_ct", "mir_9_ct", "mir_let7e_ct",
    "csf_available", "plasma_available", "serum_available",
    "saliva_available", "genetics_available", "mirna_available",
    "proteomics_available", "lipidomics_available", "rt_quic_available",
    "amyloid_beta_42_pg_ml_missing", "amyloid_beta_42_40_ratio_missing",
    "t_tau_pg_ml_missing", "p_tau181_pg_ml_missing", "nfl_pg_ml_missing",
    "gfap_pg_ml_missing", "hemoglobin_ng_ml_missing", "gdf15_pg_ml_missing",
    "klotho_pg_ml_missing", "crp40_copy_number_missing",
    "alpha_synuclein_seed_auc_missing", "alpha_synuclein_t50_hours_missing",
    "alpha_synuclein_rt_quic_result_missing", "ceramide_total_nmol_ml_missing",
    "glucosylceramide_total_nmol_ml_missing",
    "lactosylceramide_total_nmol_ml_missing", "sphingomyelin_total_nmol_ml_missing",
    "tryptophan_nm_missing", "kynurenine_nm_missing", "kynurenic_acid_nm_missing",
    "quinolinic_acid_nm_missing", "serotonin_nm_missing",
    "dopamine_mrm_intensity_missing", "creatine_mrm_intensity_missing",
    "olink_mdk_npx_missing", "olink_il17d_npx_missing", "olink_ddc_npx_missing",
    "olink_mad5_npx_missing", "mir_107_ct_missing", "mir_132_ct_missing",
    "mir_146a_ct_missing", "mir_155_ct_missing", "mir_9_ct_missing",
    "mir_let7e_ct_missing", "apoe_genotype_missing",
    "gba_mutation_carrier_missing", "gcase_activity_missing",
    "alpha_synuclein_pg_ml_missing", "mir_107_relative_expression",
    "mir_107_relative_expression_missing", "mir_132_relative_expression",
    "mir_132_relative_expression_missing", "mir_146a_relative_expression",
    "mir_146a_relative_expression_missing", "mir_155_relative_expression",
    "mir_155_relative_expression_missing", "mir_9_relative_expression",
    "mir_9_relative_expression_missing", "mir_let7e_relative_expression",
    "mir_let7e_relative_expression_missing",
]

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS  # full ordered list

# ── Label mappings ────────────────────────────────────────────────────────────
DISEASE_FULL_NAMES = {
    "AD":      "Alzheimer's Disease",
    "PD":      "Parkinson's Disease",
    "MS":      "Multiple Sclerosis",
    "Control": "No Neurological Disease (Control)",
}

RISK_LABEL_ORDER   = {"Low": 0, "Medium": 1, "High": 2}
RISK_CLASSES_LIST  = ["Low", "Medium", "High"]   # index → label

# ── Feature type mapping (for important_features in response) ─────────────────
FEATURE_TYPE_MAP = {
    "updrs_part_i":   "parkinson_clinical_assessment",
    "updrs_part_ii":  "parkinson_clinical_assessment",
    "updrs_part_iii": "parkinson_clinical_assessment",
    "updrs_part_iv":  "parkinson_clinical_assessment",
    "schwab_england_adl":       "parkinson_clinical_assessment",
    "moca_total_score":         "cognitive_assessment",
    "rem_sleep_score":          "sleep_assessment",
    "disease_duration_years":   "clinical_history",
    "levodopa_use":             "clinical_history",
    "family_history_pd":        "clinical_history",
    "age":                      "demographics",
    "sex":                      "demographics",
    "education_years":          "demographics",
    "bmi":                      "vitals",
    "systolic_bp":              "vitals",
    "diastolic_bp":             "vitals",
    "amyloid_beta_42_pg_ml":    "csf_biomarker",
    "amyloid_beta_42_40_ratio": "csf_biomarker",
    "t_tau_pg_ml":              "csf_biomarker",
    "p_tau181_pg_ml":           "csf_biomarker",
    "nfl_pg_ml":                "neurodegeneration_biomarker",
    "gfap_pg_ml":               "neurodegeneration_biomarker",
    "alpha_synuclein_pg_ml":    "synuclein_biomarker",
    "alpha_synuclein_seed_auc": "synuclein_biomarker",
    "alpha_synuclein_t50_hours":"synuclein_biomarker",
    "alpha_synuclein_rt_quic_result": "synuclein_biomarker",
    "hemoglobin_ng_ml":         "blood_biomarker",
    "gdf15_pg_ml":              "blood_biomarker",
    "klotho_pg_ml":             "blood_biomarker",
    "gcase_activity":           "enzymatic_activity",
    "beta_hexosaminidase_activity": "enzymatic_activity",
    "cathepsin_d_activity":     "enzymatic_activity",
    "apoe_genotype":            "genetics",
    "gba_mutation_carrier":     "genetics",
    "crp40_copy_number":        "genetics",
}

FEATURE_TYPE_DEFAULT = "biomarker"

# ── API settings ──────────────────────────────────────────────────────────────
API_TITLE       = "BIO_BACKEND — Neurological Risk Assessment API"
API_VERSION     = "1.0.0"
API_DESCRIPTION = (
    "FastAPI backend for multi-disease neurological classification "
    "(AD / PD / MS / Control) and risk stratification (Low / Medium / High). "
    "Powered by XGBoost + LightGBM ensemble with SHAP explainability."
)

TOP_N_FEATURES  = 5    # number of top features returned in response
DISCLAIMER_TEXT = (
    "This is a clinical decision-support output, not a final medical diagnosis."
)
RECOMMENDATION_TEXT = (
    "Recommended action: Refer the patient for detailed neurological evaluation. "
    "Further assessment may include cognitive testing, neurological examination, "
    "MRI/EEG review, repeated biomarker testing, and specialist consultation. "
    "This output should be used only as a clinical decision-support result, "
    "not as a final diagnosis."
)
