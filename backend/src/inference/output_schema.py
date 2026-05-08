from __future__ import annotations

CLINICAL_DISCLAIMER = (
    "This output is for research and decision-support demonstration only. "
    "It is not a clinical diagnosis and must not be used as a final medical decision."
)


def risk_level(ad_probability: float) -> str:
    if ad_probability <= 0.39:
        return "Low"
    if ad_probability <= 0.69:
        return "Medium"
    return "High"
