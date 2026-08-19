def build_observed_issues(predicted_class: str, risk_level: str) -> list[str]:
    if predicted_class == "Healthy":
        if risk_level == "low":
            return ["No elevated neurological speech-risk pattern was detected."]
        return ["The model output is mixed and should be interpreted cautiously."]

    labels = {
        "AD": "speech-feature pattern associated with Alzheimer's disease risk",
        "PD": "speech-feature pattern associated with Parkinson's disease risk",
        "MS": "speech-feature pattern associated with multiple sclerosis risk",
    }
    issue = labels.get(predicted_class, "neurological speech-risk pattern")
    return [f"Detected a {risk_level} {issue}."]


def build_recommendations(predicted_class: str, risk_level: str) -> list[str]:
    common = [
        "Review the result with a qualified healthcare professional.",
        "Repeat the assessment if recording quality or speech conditions were poor.",
    ]

    if predicted_class == "Healthy" and risk_level == "low":
        return [
            "Continue routine health monitoring.",
            "Seek clinical advice if new or worsening speech, memory, movement, or sensory symptoms appear.",
        ]

    if risk_level == "high":
        return [
            "Arrange a clinical evaluation with a neurologist or relevant specialist.",
            "Share the speech assessment and symptom history with the clinician.",
            *common,
        ]
    if risk_level == "moderate":
        return [
            "Monitor symptoms and consider a follow-up screening.",
            *common,
        ]
    return common
