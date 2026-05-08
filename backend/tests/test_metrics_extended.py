from __future__ import annotations

import json

import numpy as np

from src.evaluation.metrics import aggregate_fold_metrics, binary_metrics, embedding_silhouette


def _fold_metrics(acc: float, f1: float, auc: float | None) -> dict:
    return {
        "threshold": 0.5,
        "accuracy": acc,
        "balanced_accuracy": acc,
        "f1": f1,
        "precision": f1,
        "recall": f1,
        "specificity": f1,
        "auc": auc,
        "confusion_matrix": [[5, 0], [0, 5]],
    }


def test_aggregate_fold_metrics_mean_std():
    aggregate = aggregate_fold_metrics([
        _fold_metrics(0.8, 0.75, 0.85),
        _fold_metrics(0.9, 0.88, 0.92),
        _fold_metrics(0.85, 0.82, 0.88),
    ])
    assert aggregate["accuracy"]["mean"] == np.mean([0.8, 0.9, 0.85])
    assert aggregate["auc"]["valid_folds"] == 3


def test_aggregate_fold_metrics_skips_none_auc():
    aggregate = aggregate_fold_metrics([
        _fold_metrics(0.8, 0.75, None),
        _fold_metrics(0.9, 0.88, 0.92),
    ])
    assert aggregate["auc"]["valid_folds"] == 1
    assert aggregate["auc"]["mean"] == 0.92


def test_embedding_silhouette_returns_float():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(20, 256))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = [0] * 10 + [1] * 10
    score = embedding_silhouette(embeddings, labels)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_embedding_silhouette_returns_none_single_class():
    assert embedding_silhouette(np.random.randn(5, 256), [0, 0, 0, 0, 0]) is None


def test_auc_none_is_json_safe():
    metrics = binary_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert metrics["auc"] is None
    assert json.loads(json.dumps(metrics))["auc"] is None
