from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.plots import save_epoch_distribution, save_probability_bar
from src.inference.generate_zeeg import aggregate_embedding
from src.inference.report_generator import build_prediction_report, save_report
from src.models.eeg_encoder import EEGEncoderModel
from src.preprocessing.preprocess_subject import preprocess_subject
from src.training.train_eegnet import train
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _load_decision_threshold(cfg: dict, fold: int) -> tuple[float, str]:
    default = float(cfg["inference"].get("decision_threshold", 0.5))
    if not bool(cfg["inference"].get("use_calibrated_threshold", True)):
        return default, "config"

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    calibration_file = reports_dir / f"fold{fold}_threshold_calibration.json"
    if not calibration_file.exists():
        return default, "config_missing_calibration"

    try:
        with Path(calibration_file).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return float(payload["threshold"]), str(calibration_file)
    except Exception as exc:
        LOGGER.warning("Failed to load calibrated threshold from %s: %s", calibration_file, exc)
        return default, "config_invalid_calibration"


def _checkpoint_model(payload: dict, cfg: dict, epochs: np.ndarray) -> EEGEncoderModel:
    model_state = payload.get("model_state", payload)
    model_cfg = payload.get("config", cfg)
    n_channels = int(payload.get("n_channels", epochs.shape[1]))
    n_samples = int(payload.get("n_samples", epochs.shape[2]))
    expected_shape = (n_channels, n_samples)
    actual_shape = (int(epochs.shape[1]), int(epochs.shape[2]))
    if actual_shape != expected_shape:
        raise ValueError(
            "Preprocessed subject shape does not match the trained checkpoint: "
            f"expected channels/samples={expected_shape}, got {actual_shape}. "
            "Re-run preprocessing and training with the same channel and epoch configuration."
        )
    model = EEGEncoderModel(n_channels, n_samples, model_cfg)
    model.load_state_dict(model_state)
    model.eval()
    return model


def predict_subject(
    subject_id: str,
    config_path: str = "configs/training.yaml",
    auto_train: bool = True,
) -> dict:
    cfg = load_config(config_path)
    fold = int(cfg["data"].get("active_fold", 0))
    checkpoint = resolve_path(cfg["paths"]["checkpoints_dir"]) / f"eegnet_fold{fold}_best.pth"
    if not checkpoint.exists():
        if not auto_train:
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. "
                "Run python -m src.training.train_eegnet --config configs/training.yaml before launching the demo."
            )
        train(config_path, fold_override=fold)

    prep = preprocess_subject(subject_id, config_path)
    epoch_file = prep["epoch_file"]
    with np.load(epoch_file, allow_pickle=True) as data:
        epochs = data["epochs"].astype("float32")
    if epochs.shape[0] == 0:
        raise ValueError(f"No clean epochs available for {subject_id}.")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _checkpoint_model(payload, cfg, epochs)

    x = torch.from_numpy(epochs).unsqueeze(1)
    with torch.no_grad():
        out = model(x)
    epoch_probs = out["ad_probability"].cpu().numpy().astype(float).tolist()
    ad_probability = float(np.mean(epoch_probs))
    decision_threshold, threshold_source = _load_decision_threshold(cfg, fold)

    # Fix: aggregate_embedding now returns (z_eeg, consistency_score)
    z_eeg, embedding_consistency = aggregate_embedding(out["z_eeg"])
    z_eeg_list = z_eeg.tolist()

    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    prob_chart = save_probability_bar(
        1.0 - ad_probability,
        ad_probability,
        figures_dir / f"{subject_id}_probability_bar_chart.png",
    )
    dist_chart = save_epoch_distribution(
        epoch_probs,
        figures_dir / f"{subject_id}_epoch_probability_distribution.png",
    )
    visual_outputs = {
        **prep["visual_outputs"],
        "probability_bar_chart": prob_chart,
        "epoch_probability_distribution": dist_chart,
    }

    report = build_prediction_report(
        subject_id=subject_id,
        input_metadata=prep["input_metadata"],
        preprocessing_summary=prep["preprocessing_summary"],
        ad_probability=ad_probability,
        epoch_probs=epoch_probs,
        z_eeg=z_eeg_list,
        embedding_consistency=embedding_consistency,
        visual_outputs=visual_outputs,
        decision_threshold=decision_threshold,
        uncertainty_margin=float(cfg["inference"].get("uncertainty_margin", 0.05)),
        threshold_source=threshold_source,
        model_artifact={
            "checkpoint": str(checkpoint),
            "fold": fold,
            "model_version": str(cfg.get("model", {}).get("version", "phase1_eegnet_v1")),
            "model_name": str(cfg.get("model", {}).get("name", "EEGNet-Baseline")),
        },
    )
    save_report(report, reports_dir / f"{subject_id}_prediction_report.json")

    embeddings_dir = resolve_path(cfg["paths"]["embeddings_dir"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / f"{subject_id}_z_eeg.npy", np.asarray(z_eeg_list, dtype="float32"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--subject_id", default="sub-001")
    parser.add_argument("--no-auto-train", action="store_true")
    args = parser.parse_args()
    report = predict_subject(args.subject_id, args.config, auto_train=not args.no_auto_train)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
