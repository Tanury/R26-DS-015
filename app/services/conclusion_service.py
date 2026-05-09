from app.services.feature_service import format_feature_name


def generate_final_conclusion(
    predicted_disease: str,
    disease_full_name: str,
    predicted_risk: str,
    risk_confidence: float,
    important_features: list,
) -> dict:
    top_features = (
        ", ".join([format_feature_name(f["feature"]) for f in important_features[:5]])
        if important_features
        else "available clinical and biomarker features"
    )

    risk = str(predicted_risk).strip().lower()

    if predicted_disease == "Control":
        conclusion = (
            f"The clinical and biomarker profile is most consistent with a Control pattern "
            f"and indicates a {predicted_risk.upper()} neurological risk level with "
            f"{risk_confidence:.2f}% model confidence. The prediction is mainly influenced by "
            f"{top_features}."
        )

        recommendation = (
            "Recommended action: Continue routine monitoring if clinically required. "
            "If symptoms are present, further clinical evaluation is still recommended. "
            "This output should be used only as a clinical decision-support result, "
            "not as a final diagnosis."
        )

    elif risk == "high":
        conclusion = (
            f"The clinical and biomarker profile is most consistent with {disease_full_name} "
            f"and indicates a HIGH neurological risk pattern with {risk_confidence:.2f}% "
            f"model confidence. The prediction is mainly influenced by {top_features}."
        )

        recommendation = (
            "Recommended action: Refer the patient for detailed neurological evaluation. "
            "Further assessment may include cognitive testing, neurological examination, "
            "MRI/EEG review, repeated biomarker testing, and specialist consultation. "
            "This output should be used only as a clinical decision-support result, "
            "not as a final diagnosis."
        )

    elif risk == "medium":
        conclusion = (
            f"The clinical and biomarker profile is most consistent with {disease_full_name} "
            f"and indicates a MODERATE neurological risk pattern with {risk_confidence:.2f}% "
            f"model confidence. The prediction is mainly influenced by {top_features}."
        )

        recommendation = (
            "Recommended action: Continue monitoring and consider follow-up clinical assessment. "
            "The result should be reviewed together with symptoms, cognitive screening, biomarker "
            "testing, and imaging or EEG findings where available. This output should be used only "
            "as a clinical decision-support result, not as a final diagnosis."
        )

    else:
        conclusion = (
            f"The clinical and biomarker profile is most consistent with {disease_full_name} "
            f"and indicates a LOW neurological risk pattern with {risk_confidence:.2f}% "
            f"model confidence. The prediction is mainly influenced by {top_features}."
        )

        recommendation = (
            "Recommended action: Continue routine monitoring if clinically required. "
            "If symptoms are present, further clinical evaluation is still recommended. "
            "This output should be used only as a clinical decision-support result, "
            "not as a final diagnosis."
        )

    return {
        "conclusion": conclusion,
        "clinical_recommendation": recommendation,
    }