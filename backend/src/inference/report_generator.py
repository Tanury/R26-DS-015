from __future__ import annotations

import json
from pathlib import Path

from src.inference.output_schema import CLINICAL_DISCLAIMER, risk_level


def build_prediction_report(
    subject_id: str,
    input_metadata: dict,
    preprocessing_summary: dict,
    ad_probability: float,
    epoch_probs: list[float],
    z_eeg: list[float],
    embedding_consistency: float = 1.0,
    visual_outputs: dict | None = None,
    evaluation_protocol: dict | None = None,
    model_artifact: dict | None = None,
    decision_threshold: float = 0.5,
    uncertainty_margin: float = 0.05,
    threshold_source: str = "config",
) -> dict:
    hc_probability = 1.0 - ad_probability
    predicted_label = int(ad_probability >= decision_threshold)
    margin_from_threshold = abs(float(ad_probability) - float(decision_threshold))
    is_uncertain = margin_from_threshold < float(uncertainty_margin)
    prediction = "Alzheimer's EEG Pattern" if predicted_label == 1 else "Healthy Control"
    display_prediction = f"Uncertain - leaning {prediction}" if is_uncertain else prediction
    z_norm = sum(v * v for v in z_eeg) ** 0.5

    return {
        "dataset": {
            "name": "OpenNeuro ds004504",
            "stage": "Phase 1 - Binary AD vs HC",
            "training_classes": {"0": "Healthy Control", "1": "Alzheimer's EEG Pattern"},
            "excluded_groups": [
                {
                    "code": "F",
                    "label": "Frontotemporal Dementia",
                    "reason": "Excluded from Phase 1 because FTD and AD are clinically distinct.",
                }
            ],
        },
        "input_metadata": {
            **input_metadata,
            "real_data": not bool(input_metadata.get("synthetic_data", False)),
        },
        "preprocessing_summary": preprocessing_summary,
        "model_summary": {
            "model": "EEGNet-Baseline",
            "task": "Alzheimer's EEG Pattern vs Healthy Control",
            "classification_type": "Binary classification",
            "analysis_level": "Subject-level prediction",
            "embedding_dimension": 256,
            "embedding_normalization": "L2-normalized",
        },
        "model_artifact": model_artifact or {},
        "evaluation_protocol": evaluation_protocol or {
            "split_strategy": "Subject-level split",
            "cross_validation": "StratifiedGroupKFold",
            "group_key": "subject_id",
            "no_epoch_leakage": True,
            "important_note": "Epochs from the same subject are never placed in both training and testing sets.",
        },
        "subject_level_prediction": {
            "prediction": display_prediction,
            "leaning_prediction": prediction,
            "predicted_label": predicted_label,
            "is_uncertain": is_uncertain,
            "ad_eeg_pattern_probability": round(float(ad_probability), 4),
            "predicted_class_confidence": round(float(max(ad_probability, hc_probability)), 4),
            "subject_level_confidence": round(float(max(ad_probability, hc_probability)), 4),
            "confidence_interpretation": (
                "Aggregated model probability, not clinical certainty. "
                "Predictions inside the uncertainty margin should be treated as inconclusive."
            ),
            "decision_threshold": round(float(decision_threshold), 4),
            "threshold_source": threshold_source,
            "uncertainty_margin": round(float(uncertainty_margin), 4),
            "margin_from_threshold": round(float(margin_from_threshold), 4),
            "risk_level": risk_level(float(ad_probability)),
            "aggregation_method": "Mean probability across clean epochs",
            "class_probabilities": {
                "Healthy Control": round(float(hc_probability), 4),
                "Alzheimer's EEG Pattern": round(float(ad_probability), 4),
            },
        },
        "epoch_probability_summary": {
            "epochs_used_for_prediction": len(epoch_probs),
            "mean_ad_probability": round(float(ad_probability), 4),
            "std_ad_probability": round(float(_safe_std(epoch_probs)), 4),
            "min_ad_probability": round(float(min(epoch_probs)), 4) if epoch_probs else None,
            "max_ad_probability": round(float(max(epoch_probs)), 4) if epoch_probs else None,
            "note": "Epoch probabilities are summarized only to explain prediction consistency across EEG windows.",
        },
        "bar_chart_data": [
            {
                "class": "Healthy Control",
                "probability": round(float(hc_probability), 4),
                "percentage": f"{hc_probability:.0%}",
            },
            {
                "class": "Alzheimer's EEG Pattern",
                "probability": round(float(ad_probability), 4),
                "percentage": f"{ad_probability:.0%}",
            },
        ],
        "embedding_output": {
            "z_eeg_shape": [256],
            "l2_norm": round(float(z_norm), 4),
            "availability_flag": 1,
            # Fix: expose embedding consistency — 1.0 means all epochs fully aligned,
            # lower means higher variance across epoch embeddings.
            "embedding_consistency": round(float(embedding_consistency), 4),
            "z_eeg_preview": [round(float(v), 3) for v in z_eeg[:4]] + ["..."],
            "note": (
                "Only the first few embedding values are shown for display; "
                "the full vector contains 256 values. "
                "embedding_consistency near 1.0 indicates stable, consistent epoch embeddings."
            ),
        },
        "visual_outputs": visual_outputs or {},
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
    }


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def save_report(report: dict, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return str(p)
