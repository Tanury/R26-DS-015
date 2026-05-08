from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EpochDataset, load_split
from src.evaluation.leakage_check import check_no_subject_leakage
from src.evaluation.metrics import aggregate_fold_metrics, binary_metrics, embedding_silhouette, find_best_threshold
from src.evaluation.plots import (
    save_confusion_matrix,
    save_embedding_scatter,
    save_roc_curve,
)
from src.models.eeg_encoder import EEGEncoderModel
from src.training.train_eegnet import train
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.file_utils import write_json


def _first_epoch_shape(epochs_dir: Path) -> tuple[int, int]:
    files = sorted(epochs_dir.glob("*_epochs.npz"))
    if not files:
        raise FileNotFoundError(f"No epoch files found in {epochs_dir}. Run preprocessing first.")
    with np.load(files[0], allow_pickle=True) as data:
        _, channels, samples = data["epochs"].shape
    return int(channels), int(samples)


def _checkpoint_model(payload: dict, cfg: dict, epochs_dir: Path) -> EEGEncoderModel:
    model_state = payload.get("model_state", payload)
    model_cfg = payload.get("config", cfg)
    if "n_channels" in payload and "n_samples" in payload:
        n_channels = int(payload["n_channels"])
        n_samples = int(payload["n_samples"])
    else:
        n_channels, n_samples = _first_epoch_shape(epochs_dir)
    model = EEGEncoderModel(n_channels, n_samples, model_cfg)
    model.load_state_dict(model_state)
    model.eval()
    return model


def evaluate(
    config_path: str = "configs/training.yaml",
    fold_override: int | None = None,
    auto_train: bool = True,
    write_report: bool = True,
) -> dict:
    cfg = load_config(config_path)
    fold = int(cfg["data"].get("active_fold", 0) if fold_override is None else fold_override)
    checkpoint = resolve_path(cfg["paths"]["checkpoints_dir"]) / f"eegnet_fold{fold}_best.pth"
    if not checkpoint.exists():
        if not auto_train:
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. "
                "Train this fold before evaluation or omit --no-auto-train."
            )
        train(config_path, fold_override=fold)

    split_file = resolve_data_file(cfg, "split_file", "splits_dir")
    leak = check_no_subject_leakage(str(split_file))
    if not leak["no_epoch_leakage"]:
        raise RuntimeError("Subject leakage detected.")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epochs_dir = resolve_path(cfg["paths"]["processed_epochs_dir"])
    model = _checkpoint_model(payload, cfg, epochs_dir)

    val_meta = load_split(split_file, fold, "val")
    max_epochs = cfg["data"].get("max_eval_epochs_per_subject")
    max_epochs = None if max_epochs in (None, "null") else int(max_epochs)
    ds = EpochDataset(
        val_meta,
        epochs_dir,
        min_epochs=int(cfg["preprocessing"].get("min_clean_epochs", 1)),
        max_epochs_per_subject=max_epochs,
        seed=int(cfg["project"].get("seed", 42)),
    )
    if cfg["data"].get("fail_on_skipped_subjects", True) and ds.skipped_subjects:
        raise RuntimeError(f"Skipped validation subjects during evaluation: {sorted(ds.skipped_subjects)}")
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    by_subject = defaultdict(lambda: {"probs": [], "z": [], "label": None})
    with torch.no_grad():
        for x, y, subject_ids in loader:
            out = model(x)
            probs = out["ad_probability"].cpu().numpy().tolist()
            embeddings = out["z_eeg"].cpu().numpy()
            labels = y.cpu().numpy().tolist()
            for sid, prob, z_epoch, label in zip(subject_ids, probs, embeddings, labels):
                by_subject[sid]["probs"].append(float(prob))
                by_subject[sid]["z"].append(z_epoch.astype("float32"))
                by_subject[sid]["label"] = int(label)

    rows = []
    subject_embeddings = []
    subject_labels = []
    for subject_id, item in by_subject.items():
        z_mean = np.mean(np.vstack(item["z"]), axis=0)
        norm = float(np.linalg.norm(z_mean))
        if norm > 0:
            z_mean = z_mean / norm
        subject_embeddings.append(z_mean.astype("float32"))
        subject_labels.append(int(item["label"]))
        rows.append(
            {
                "subject_id": subject_id,
                "label": item["label"],
                "mean_ad_probability": float(np.mean(item["probs"])),
                "epochs": len(item["probs"]),
            }
        )
    if not rows:
        raise RuntimeError("No subject-level predictions were generated for evaluation.")

    subject_df = pd.DataFrame(rows).sort_values("subject_id")
    y_true = subject_df["label"].astype(int).tolist()
    y_prob = subject_df["mean_ad_probability"].tolist()
    default_threshold = float(cfg["inference"]["decision_threshold"])
    default_metrics = binary_metrics(y_true, y_prob, threshold=default_threshold)
    threshold_calibration = find_best_threshold(
        y_true,
        y_prob,
        objective=str(cfg["inference"].get("threshold_objective", "balanced_accuracy")),
    )
    calibrated_threshold = float(threshold_calibration["threshold"])
    metrics = binary_metrics(y_true, y_prob, threshold=calibrated_threshold)
    metrics["default_threshold_metrics"] = default_metrics
    metrics["threshold_calibration"] = threshold_calibration
    metrics["embedding_silhouette"] = embedding_silhouette(
        np.vstack(subject_embeddings),
        subject_labels,
    )

    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    embeddings_dir = resolve_path(cfg["paths"]["embeddings_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(reports_dir / f"fold{fold}_subject_predictions.csv", index=False)
    write_json(reports_dir / f"fold{fold}_threshold_calibration.json", threshold_calibration)
    np.save(embeddings_dir / f"fold{fold}_subject_embeddings.npy", np.vstack(subject_embeddings))
    cm_path = save_confusion_matrix(metrics["confusion_matrix"], figures_dir / f"fold{fold}_confusion_matrix.png")
    roc_path = save_roc_curve(y_true, y_prob, figures_dir / f"fold{fold}_roc_curve.png") if len(set(y_true)) == 2 else None
    pca_path = save_embedding_scatter(
        np.vstack(subject_embeddings),
        subject_labels,
        figures_dir / f"fold{fold}_embedding_pca.png",
        method="pca",
        subject_ids=subject_df["subject_id"].astype(str).tolist(),
    )
    tsne_path = None
    if len(subject_embeddings) >= 10:
        tsne_path = save_embedding_scatter(
            np.vstack(subject_embeddings),
            subject_labels,
            figures_dir / f"fold{fold}_embedding_tsne.png",
            method="tsne",
            subject_ids=subject_df["subject_id"].astype(str).tolist(),
        )
    result = {
        "model": "EEGNet-Baseline",
        "analysis_level": "Subject-level prediction",
        "fold": fold,
        "metrics": metrics,
        "leakage_check": leak,
        "visual_outputs": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "embedding_pca": pca_path,
            "embedding_tsne": tsne_path,
        },
    }
    if write_report:
        write_json(reports_dir / "phase1_metrics.json", result)
    return result


def evaluate_all_folds(config_path: str = "configs/training.yaml", auto_train: bool = True) -> dict:
    cfg = load_config(config_path)
    n_splits = int(cfg["data"].get("n_splits", 5))
    per_fold = [
        evaluate(config_path, fold_override=fold, auto_train=auto_train, write_report=False)
        for fold in range(n_splits)
    ]
    aggregate = aggregate_fold_metrics([fold_result["metrics"] for fold_result in per_fold])
    silhouette_values = [
        fold_result["metrics"].get("embedding_silhouette")
        for fold_result in per_fold
        if fold_result["metrics"].get("embedding_silhouette") is not None
    ]
    aggregate["embedding_silhouette"] = {
        "values": silhouette_values,
        "mean": float(np.mean(silhouette_values)) if silhouette_values else None,
        "std": float(np.std(silhouette_values, ddof=0)) if silhouette_values else None,
    }
    result = {
        "model": "EEGNet-Baseline",
        "analysis_level": "Subject-level prediction",
        "cross_validation": "StratifiedGroupKFold",
        "folds_evaluated": n_splits,
        "aggregate_metrics": aggregate,
        "per_fold": per_fold,
    }
    write_json(resolve_path(cfg["paths"]["reports_dir"]) / "phase1_crossval_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--no-auto-train", action="store_true")
    args = parser.parse_args()
    if args.all_folds:
        evaluate_all_folds(args.config, auto_train=not args.no_auto_train)
    else:
        evaluate(args.config, fold_override=args.fold, auto_train=not args.no_auto_train)


if __name__ == "__main__":
    main()
