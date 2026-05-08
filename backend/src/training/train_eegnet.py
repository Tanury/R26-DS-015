from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import EpochDataset, load_split
from src.data.create_subject_splits import create_subject_splits
from src.evaluation.metrics import binary_metrics
from src.evaluation.validate_preprocessed_shapes import validate_preprocessed_shapes
from src.models.eeg_encoder import EEGEncoderModel
from src.preprocessing.preprocess_subject import preprocess_all
from src.training.callbacks import EarlyStopping
from src.training.losses import class_weights_from_labels
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.file_utils import write_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _device(cfg: dict) -> torch.device:
    requested = cfg["training"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _ensure_synthetic_labels(cfg: dict) -> None:
    label_file = resolve_data_file(cfg, "label_file", "labels_dir")
    if label_file.exists():
        return
    if not cfg["data"].get("synthetic_fallback", False):
        raise FileNotFoundError(
            f"Missing label file: {label_file}. "
            "Run: python -m src.data.prepare_labels --config configs/training.yaml"
        )
    if not cfg["data"].get("synthetic_fallback", True):
        return
    label_file.parent.mkdir(parents=True, exist_ok=True)
    synthetic_label_file = label_file.parent / "synthetic_ad_hc_subjects.csv"
    n = int(cfg["data"].get("synthetic_subjects_per_class", 6))
    rows = []
    for i in range(n):
        rows.append({
            "subject_id": f"sub-synth-hc-{i:03d}",
            "participant_id": f"sub-synth-hc-{i:03d}",
            "group_code": "C",
            "label": 0,
            "label_name": "Healthy Control",
        })
        rows.append({
            "subject_id": f"sub-synth-ad-{i:03d}",
            "participant_id": f"sub-synth-ad-{i:03d}",
            "group_code": "A",
            "label": 1,
            "label_name": "Alzheimer's EEG Pattern",
        })
    pd.DataFrame(rows).to_csv(label_file, index=False)
    pd.DataFrame(rows).to_csv(synthetic_label_file, index=False)
    LOGGER.warning(
        "SYNTHETIC label file created (%d HC + %d AD). "
        "No real dataset found — results are NOT clinically valid.",
        n, n,
    )


def _first_epoch_shape(epochs_dir: Path) -> tuple[int, int]:
    files = sorted(epochs_dir.glob("*_epochs.npz"))
    if not files:
        raise FileNotFoundError(
            f"No epoch files found in {epochs_dir}. Run preprocessing first."
        )
    with np.load(files[0], allow_pickle=True) as data:
        _, channels, samples = data["epochs"].shape
    return int(channels), int(samples)


def _subject_metrics(probs: list[float], labels: list[int], subject_ids: list[str]) -> dict:
    by_subject: dict[str, dict[str, list[float] | int]] = {}
    for prob, label, subject_id in zip(probs, labels, subject_ids):
        item = by_subject.setdefault(subject_id, {"probs": [], "label": int(label)})
        item["probs"].append(float(prob))  # type: ignore[index]
    y_true = [int(item["label"]) for item in by_subject.values()]
    y_prob = [float(np.mean(item["probs"])) for item in by_subject.values()]  # type: ignore[arg-type]
    return binary_metrics(y_true, y_prob)


def _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip_norm: float | None = None):
    training = optimizer is not None
    model.train(training)
    losses, probs, labels, subject_ids_all = [], [], [], []
    for x, y, subject_ids in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            out = model(x)
            loss = criterion(out["logits"], y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        probs.extend(out["ad_probability"].detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
        subject_ids_all.extend([str(sid) for sid in subject_ids])
    epoch_preds = [1 if p >= 0.5 else 0 for p in probs]
    epoch_auc = float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None
    subject = _subject_metrics(probs, labels, subject_ids_all)
    return {
        "loss": float(np.mean(losses)),
        "epoch_f1": float(f1_score(labels, epoch_preds, zero_division=0)),
        "epoch_auc": epoch_auc,
        "subject_f1": subject["f1"],
        "subject_auc": subject["auc"],
        "subject_accuracy": subject["accuracy"],
    }


def train(config_path: str = "configs/training.yaml", fold_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    if fold_override is not None:
        cfg["data"]["active_fold"] = int(fold_override)
    set_seed(int(cfg["project"]["seed"]))
    _ensure_synthetic_labels(cfg)

    if not list(resolve_path(cfg["paths"]["processed_epochs_dir"]).glob("*_epochs.npz")):
        preprocess_all(config_path)
    shape_validation = validate_preprocessed_shapes(config_path)
    if not shape_validation["passed"]:
        raise RuntimeError(
            "Preprocessed epoch validation failed. "
            "Run preprocessing again, or use --force with preprocessing before training."
        )
    if cfg["data"].get("recreate_splits_on_train", True) or not resolve_data_file(cfg, "split_file", "splits_dir").exists():
        create_subject_splits(config_path)

    device = _device(cfg)
    fold = int(cfg["data"].get("active_fold", 0))
    split_file = resolve_data_file(cfg, "split_file", "splits_dir")
    epochs_dir = resolve_path(cfg["paths"]["processed_epochs_dir"])
    checkpoints_dir = resolve_path(cfg["paths"]["checkpoints_dir"])
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_meta = load_split(split_file, fold, "train")
    val_meta = load_split(split_file, fold, "val")
    min_epochs = int(cfg["preprocessing"].get("min_clean_epochs", 1))
    max_epochs = cfg["data"].get("max_train_epochs_per_subject", cfg["data"].get("max_epochs_per_subject"))
    max_epochs = None if max_epochs in (None, "null") else int(max_epochs)
    seed = int(cfg["project"].get("seed", 42))
    train_ds = EpochDataset(
        train_meta,
        epochs_dir,
        min_epochs=min_epochs,
        max_epochs_per_subject=max_epochs,
        seed=seed,
    )
    val_ds = EpochDataset(
        val_meta,
        epochs_dir,
        min_epochs=min_epochs,
        max_epochs_per_subject=max_epochs,
        seed=seed,
    )
    if cfg["data"].get("fail_on_skipped_subjects", True):
        skipped = sorted(set(train_ds.skipped_subjects + val_ds.skipped_subjects))
        if skipped:
            raise RuntimeError(f"Skipped subjects during dataset loading: {skipped}")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    n_channels, n_samples = _first_epoch_shape(epochs_dir)
    model = EEGEncoderModel(n_channels, n_samples, cfg).to(device)

    weights = None
    if cfg["training"].get("use_class_weights", True):
        weights = class_weights_from_labels(
            train_meta["label"].astype(int).tolist()
        ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    # LR scheduler: reduce on plateau to escape flat regions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=int(cfg["training"].get("lr_patience", 3)),
    )

    # Fix: EarlyStopping.step() now returns (should_stop, is_best) — single source of truth
    stopper = EarlyStopping(int(cfg["training"]["early_stopping_patience"]))
    history = []
    best_metrics = None
    best_path = checkpoints_dir / f"eegnet_fold{fold}_best.pth"
    grad_clip_norm = cfg["training"].get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm in (None, "null") else float(grad_clip_norm)

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer, grad_clip_norm)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        val_auc = "undefined" if val_metrics["subject_auc"] is None else f"{val_metrics['subject_auc']:.4f}"
        LOGGER.info(
            "Epoch %s | train loss %.4f | val loss %.4f | subject val auc %s",
            epoch, train_metrics["loss"], val_metrics["loss"], val_auc,
        )

        scheduler.step(val_metrics["loss"])
        monitor_name = "subject_auc" if val_metrics["subject_auc"] is not None else "val_loss"
        monitor_value = (
            -float(val_metrics["subject_auc"])
            if val_metrics["subject_auc"] is not None
            else float(val_metrics["loss"])
        )

        # Prefer subject-level AUC for checkpoint selection when defined; fall back to validation loss.
        should_stop, is_best = stopper.step(monitor_value)
        if is_best:
            best_metrics = row
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "n_channels": n_channels,
                    "n_samples": n_samples,
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_auc": val_metrics["subject_auc"],
                    "monitor": monitor_name,
                    "monitor_value": monitor_value,
                },
                best_path,
            )
            LOGGER.info("New best checkpoint saved (%s=%.4f)", monitor_name, -monitor_value if monitor_name == "subject_auc" else monitor_value)
        if should_stop and cfg["training"].get("enable_early_stopping", True):
            LOGGER.info("Early stopping at epoch %s", epoch)
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(reports_dir / f"fold{fold}_training_history.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["train_subject_accuracy"], label="train")
    axes[0, 1].plot(history_df["epoch"], history_df["val_subject_accuracy"], label="val")
    axes[0, 1].set_title("Subject Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[1, 0].plot(history_df["epoch"], history_df["train_subject_f1"], label="train")
    axes[1, 0].plot(history_df["epoch"], history_df["val_subject_f1"], label="val")
    axes[1, 0].set_title("Subject F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    if history_df["train_subject_auc"].notna().any() or history_df["val_subject_auc"].notna().any():
        axes[1, 1].plot(history_df["epoch"], history_df["train_subject_auc"], label="train")
        axes[1, 1].plot(history_df["epoch"], history_df["val_subject_auc"], label="val")
    axes[1, 1].set_title("Subject AUC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"fold{fold}_training_curve.png", dpi=140)
    plt.close()

    summary = {
        "fold": fold,
        "best_checkpoint": str(best_path),
        "best_metrics": best_metrics or history[-1],
        "last_metrics": history[-1],
        "device": str(device),
        "overfit_controls": {
            "checkpoint_monitor": "subject_auc_when_defined_else_val_loss",
            "early_stopping_patience": int(cfg["training"]["early_stopping_patience"]),
            "dropout": float(cfg["model"].get("dropout", 0.5)),
            "weight_decay": float(cfg["training"]["weight_decay"]),
            "label_smoothing": float(cfg["training"].get("label_smoothing", 0.0)),
            "grad_clip_norm": grad_clip_norm,
            "max_train_epochs_per_subject": max_epochs,
        },
    }
    write_json(reports_dir / f"fold{fold}_training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    train(args.config, fold_override=args.fold)


if __name__ == "__main__":
    main()
