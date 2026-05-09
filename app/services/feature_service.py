def get_disease_full_name(label: str) -> str:
    mapping = {
        "AD": "Alzheimer’s Disease",
        "PD": "Parkinson’s Disease",
        "MS": "Multiple Sclerosis",
        "Control": "Control / No major neurological disease pattern",
    }

    return mapping.get(str(label), str(label))


def get_feature_type(feature_name: str) -> str:
    biomarker_features = {
        "amyloid_beta_42_pg_ml",
        "amyloid_beta_40_pg_ml",
        "p_tau181_pg_ml",
        "t_tau_pg_ml",
        "nfl_pg_ml",
        "gfap_pg_ml",
        "alpha_synuclein_pg_ml",
        "neuroinflam_score",
        "tau_amyloid_ratio",
    }

    cognitive_features = {
        "moca_total_score",
    }

    parkinson_features = {
        "updrs_part_i",
        "updrs_part_ii",
        "updrs_part_iii",
        "updrs_part_iv",
    }

    history_features = {
        "age",
        "sex",
        "disease_duration_years",
    }

    if feature_name in biomarker_features:
        return "fluid_biomarker"

    if feature_name in cognitive_features:
        return "cognitive_assessment"

    if feature_name in parkinson_features:
        return "parkinson_clinical_assessment"

    if feature_name in history_features:
        return "clinical_history"

    return "other_clinical_feature"


def format_feature_name(feature: str) -> str:
    readable_names = {
        "updrs_part_i": "UPDRS Part I",
        "updrs_part_ii": "UPDRS Part II",
        "updrs_part_iii": "UPDRS Part III",
        "updrs_part_iv": "UPDRS Part IV",
        "moca_total_score": "MoCA total score",
        "disease_duration_years": "Disease duration",
        "amyloid_beta_42_pg_ml": "Amyloid-beta 42",
        "amyloid_beta_40_pg_ml": "Amyloid-beta 40",
        "p_tau181_pg_ml": "p-tau181",
        "t_tau_pg_ml": "Total tau",
        "nfl_pg_ml": "NfL",
        "gfap_pg_ml": "GFAP",
        "alpha_synuclein_pg_ml": "Alpha-synuclein",
        "neuroinflam_score": "Neuroinflammation score",
        "tau_amyloid_ratio": "Tau-amyloid ratio",
        "age": "Age",
        "sex": "Sex",
    }

    return readable_names.get(feature, feature.replace("_", " ").title())