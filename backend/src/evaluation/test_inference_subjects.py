from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from src.models.eeg_encoder import EEGEncoderModel
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.file_utils import write_json


def _pick_subjects(labels: pd.DataFrame, n_subjects: int) -> pd.DataFrame:
    per_class = n_subjects // 2
    remainder = n_subjects - (per_class * 2)
    selected = []
    for label, count in [(1, per_class + remainder), (0, per_class)]:
        part = labels[labels["label"].eq(label)].sort_values("subject_id").head(count)
        selected.append(part)
    return pd.concat(selected, ignore_index=True).sort_values("subject_id").reset_index(drop=True)


def _load_threshold(reports_dir: Path, fold: int, default: float) -> tuple[float, str]:
    path = reports_dir / f"fold{fold}_threshold_calibration.json"
    if not path.exists():
        return default, "config"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["threshold"]), str(path)


def _load_model(checkpoint: Path, cfg: dict, epochs_shape: tuple[int, int]) -> EEGEncoderModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg = payload.get("config", cfg)
    n_channels = int(payload.get("n_channels", epochs_shape[0]))
    n_samples = int(payload.get("n_samples", epochs_shape[1]))
    model = EEGEncoderModel(n_channels, n_samples, model_cfg)
    model.load_state_dict(payload.get("model_state", payload))
    model.eval()
    return model


def _subject_split(split_df: pd.DataFrame, subject_id: str, fold: int) -> str:
    rows = split_df[(split_df["subject_id"].eq(subject_id)) & (split_df["fold"].eq(fold))]
    if rows.empty:
        return "unknown"
    return str(rows.iloc[0]["split"])


def run_test(config_path: str, n_subjects: int, folds: list[int] | None = None) -> dict:
    cfg = load_config(config_path)
    folds = folds if folds is not None else list(range(int(cfg["data"].get("n_splits", 5))))
    labels = pd.read_csv(resolve_data_file(cfg, "label_file", "labels_dir"))
    labels = labels[labels["label"].isin([0, 1])].copy()
    subjects = _pick_subjects(labels, n_subjects)
    split_df = pd.read_csv(resolve_data_file(cfg, "split_file", "splits_dir"))

    epochs_dir = resolve_path(cfg["paths"]["processed_epochs_dir"])
    checkpoints_dir = resolve_path(cfg["paths"]["checkpoints_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    default_threshold = float(cfg["inference"].get("decision_threshold", 0.5))

    rows = []
    model_cache: dict[int, EEGEncoderModel] = {}
    threshold_cache: dict[int, tuple[float, str]] = {}

    for _, subject in subjects.iterrows():
        subject_id = str(subject["subject_id"])
        label = int(subject["label"])
        epoch_path = epochs_dir / f"{subject_id}_epochs.npz"
        if not epoch_path.exists():
            rows.append(
                {
                    "subject_id": subject_id,
                    "true_label": label,
                    "true_name": subject["label_name"],
                    "status": "missing_epochs",
                }
            )
            continue
        with np.load(epoch_path, allow_pickle=True) as data:
            epochs = data["epochs"].astype("float32")
        if len(epochs) == 0:
            rows.append(
                {
                    "subject_id": subject_id,
                    "true_label": label,
                    "true_name": subject["label_name"],
                    "status": "zero_epochs",
                }
            )
            continue

        x = torch.from_numpy(epochs).unsqueeze(1)
        for fold in folds:
            checkpoint = checkpoints_dir / f"eegnet_fold{fold}_best.pth"
            if not checkpoint.exists():
                rows.append(
                    {
                        "subject_id": subject_id,
                        "fold": fold,
                        "true_label": label,
                        "true_name": subject["label_name"],
                        "status": "missing_checkpoint",
                    }
                )
                continue
            if fold not in model_cache:
                model_cache[fold] = _load_model(checkpoint, cfg, (epochs.shape[1], epochs.shape[2]))
            if fold not in threshold_cache:
                threshold_cache[fold] = _load_threshold(reports_dir, fold, default_threshold)
            threshold, threshold_source = threshold_cache[fold]
            with torch.no_grad():
                out = model_cache[fold](x)
            probs = out["ad_probability"].cpu().numpy().astype(float)
            ad_probability = float(np.mean(probs))
            predicted_label = int(ad_probability >= threshold)
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold": fold,
                    "fold_split": _subject_split(split_df, subject_id, fold),
                    "true_label": label,
                    "true_name": subject["label_name"],
                    "predicted_label": predicted_label,
                    "prediction": "Alzheimer's EEG Pattern" if predicted_label else "Healthy Control",
                    "ad_probability": ad_probability,
                    "threshold": threshold,
                    "threshold_source": threshold_source,
                    "correct": predicted_label == label,
                    "epochs": int(len(epochs)),
                    "epoch_ad_probability_std": float(np.std(probs)),
                    "status": "ok",
                }
            )

    result_df = pd.DataFrame(rows)
    ok_df = result_df[result_df["status"].eq("ok")].copy()
    overall = _metrics(ok_df)
    per_fold = {
        str(fold): _metrics(ok_df[ok_df["fold"].eq(fold)])
        for fold in folds
    }
    per_split = {
        split: _metrics(part)
        for split, part in ok_df.groupby("fold_split")
    }
    payload = {
        "config": config_path,
        "n_requested_subjects": n_subjects,
        "subjects": subjects["subject_id"].astype(str).tolist(),
        "folds": folds,
        "overall_metrics": overall,
        "per_fold_metrics": per_fold,
        "per_split_metrics": per_split,
        "rows": result_df.to_dict(orient="records"),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    result_csv = reports_dir / f"inference_10_subjects_folds_{min(folds)}_{max(folds)}.csv"
    result_json = reports_dir / f"inference_10_subjects_folds_{min(folds)}_{max(folds)}.json"
    result_df.to_csv(result_csv, index=False)
    write_json(result_json, payload)
    payload["csv_path"] = str(result_csv)
    payload["json_path"] = str(result_json)
    return payload


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    y_true = df["true_label"].astype(int).tolist()
    y_pred = df["predicted_label"].astype(int).tolist()
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--folds", nargs="*", type=int)
    args = parser.parse_args()
    result = run_test(args.config, args.n_subjects, args.folds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
