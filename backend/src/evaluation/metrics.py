from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _threshold_predictions(y_prob: list[float], threshold: float) -> list[int]:
    return [1 if p >= threshold else 0 for p in y_prob]


def binary_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    """Compute subject-level binary classification metrics.

    AUC is set to None (not NaN) when only one class is present in the
    validation fold, with an explicit warning logged. NaN silently propagates
    into JSON and breaks downstream parsing.
    """
    y_pred = _threshold_predictions(y_prob, threshold)

    if len(set(y_true)) < 2:
        LOGGER.warning(
            "AUC is undefined: only one class present in this fold's validation set "
            "(%s). This can happen with small datasets. "
            "Try increasing n_splits or using a different fold.",
            set(y_true),
        )
        auc = None
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    cm = np.asarray(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "auc": auc,  # None instead of NaN — safe for JSON serialization
        "confusion_matrix": cm.tolist(),
    }


def find_best_threshold(
    y_true: list[int],
    y_prob: list[float],
    *,
    objective: str = "balanced_accuracy",
) -> dict:
    """Find a validation threshold from subject-level probabilities.

    This is intentionally simple and transparent: candidate thresholds are the
    unique validation probabilities plus 0.5. Ties prefer the threshold closest
    to 0.5 so tiny folds do not drift more than necessary.
    """
    if not y_true or not y_prob:
        raise ValueError("Cannot calibrate threshold without validation predictions.")

    candidates = sorted({0.5, *[float(p) for p in y_prob]})
    best: dict | None = None
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_prob, threshold=threshold)
        score = float(metrics.get(objective, metrics["balanced_accuracy"]))
        item = {
            "threshold": float(threshold),
            "objective": objective,
            "objective_value": score,
            "metrics": metrics,
        }
        if best is None:
            best = item
            continue
        if score > float(best["objective_value"]):
            best = item
        elif score == float(best["objective_value"]) and abs(threshold - 0.5) < abs(float(best["threshold"]) - 0.5):
            best = item

    assert best is not None
    return best


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold subject-level metrics into mean/std values."""
    summary: dict = {}
    for key in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]:
        values = [float(item[key]) for item in fold_metrics if item.get(key) is not None]
        summary[key] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
        }

    auc_values = [float(item["auc"]) for item in fold_metrics if item.get("auc") is not None]
    summary["auc"] = {
        "values": auc_values,
        "mean": float(np.mean(auc_values)) if auc_values else None,
        "std": float(np.std(auc_values, ddof=0)) if auc_values else None,
        "valid_folds": len(auc_values),
    }
    return summary


def embedding_silhouette(embeddings: np.ndarray, labels: list[int]) -> float | None:
    """Compute a cosine silhouette score for subject-level z_eeg embeddings."""
    if len(embeddings) < 2 or len(set(labels)) < 2:
        LOGGER.warning(
            "Silhouette score requires at least 2 subjects and 2 classes. Got %d subjects and %d classes.",
            len(embeddings),
            len(set(labels)),
        )
        return None
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(embeddings, labels, metric="cosine"))
    except Exception as exc:
        LOGGER.warning("Embedding silhouette computation failed: %s", exc)
        return None
