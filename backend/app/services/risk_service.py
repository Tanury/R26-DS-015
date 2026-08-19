NON_RISK_CLASS = "Healthy"


def calculate_risk_score(predicted_class: str, confidence: float) -> float:
    if predicted_class == NON_RISK_CLASS:
        return round(1.0 - confidence, 6)
    return round(confidence, 6)


def derive_risk_level(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "high"
    if risk_score >= 0.4:
        return "moderate"
    return "low"
