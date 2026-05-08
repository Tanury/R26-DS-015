from __future__ import annotations

from src.evaluation.metrics import binary_metrics
from src.utils.config_loader import load_config


def test_training_config_has_runtime_artifact_paths():
    cfg = load_config("configs/training.yaml")
    for key in ["checkpoints_dir", "embeddings_dir", "figures_dir", "reports_dir"]:
        assert key in cfg["paths"], f"Missing runtime path: {key}"
    assert cfg["data"]["synthetic_fallback"] is False
    assert cfg["data"]["fail_on_skipped_subjects"] is True


def test_synthetic_test_config_enables_synthetic_fallback():
    cfg = load_config("configs/training_synthetic_test.yaml")
    assert cfg["data"]["synthetic_fallback"] is True
    assert "dataset_root" in cfg["paths"]
    assert "processed_epochs_dir" in cfg["paths"]
    assert cfg["preprocessing"]["low_freq_hz"] == 0.5
    assert len(cfg["preprocessing"]["selected_channels"]) == 19
    assert cfg["training"]["epochs"] == 2


def test_training_config_inherits_preprocessing_settings():
    cfg = load_config("configs/training.yaml")
    assert cfg["preprocessing"]["low_freq_hz"] == 0.5
    assert cfg["preprocessing"]["high_freq_hz"] == 40.0


def test_auc_is_none_when_only_one_class_present():
    metrics = binary_metrics([1, 1, 1], [0.8, 0.7, 0.9])
    assert metrics["auc"] is None


def test_training_config_uses_calibrated_thresholds():
    cfg = load_config("configs/training.yaml")
    assert cfg["inference"]["use_calibrated_threshold"] is True
    assert cfg["inference"]["threshold_objective"] == "balanced_accuracy"
    assert cfg["inference"]["uncertainty_margin"] == 0.05
