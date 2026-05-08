from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.evaluate_subject_level import evaluate, evaluate_all_folds
from src.evaluation.leakage_check import check_no_subject_leakage
from src.evaluation.validate_preprocessed_shapes import validate_preprocessed_shapes
from src.inference.output_schema import CLINICAL_DISCLAIMER
from src.inference.predict_subject import predict_subject
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.file_utils import write_json


class ModalCheck:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def ok(self, name: str, details: dict[str, Any] | None = None) -> None:
        self.results.append({"name": name, "passed": True, "details": details or {}})

    def fail(self, name: str, message: str, details: dict[str, Any] | None = None) -> None:
        payload = {"message": message}
        payload.update(details or {})
        self.results.append({"name": name, "passed": False, "details": payload})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.results)


def _check_file(check: ModalCheck, name: str, path: Path) -> bool:
    if path.exists():
        check.ok(name, {"path": str(path)})
        return True
    check.fail(name, "Required file is missing.", {"path": str(path)})
    return False


def _check_dir_has_files(check: ModalCheck, name: str, path: Path, pattern: str) -> bool:
    files = sorted(path.glob(pattern)) if path.exists() else []
    if files:
        check.ok(name, {"path": str(path), "pattern": pattern, "count": len(files)})
        return True
    check.fail(name, "Required directory has no matching files.", {"path": str(path), "pattern": pattern})
    return False


def _validate_labels(check: ModalCheck, label_file: Path) -> None:
    if not _check_file(check, "label_file_exists", label_file):
        return
    labels = pd.read_csv(label_file)
    counts = labels["label"].astype(int).value_counts().to_dict()
    hc = int(counts.get(0, 0))
    ad = int(counts.get(1, 0))
    if hc > 0 and ad > 0 and set(labels["label"].astype(int).unique()).issubset({0, 1}):
        check.ok("label_file_valid_ad_hc", {"healthy_control": hc, "ad_eeg_pattern": ad})
    else:
        check.fail(
            "label_file_valid_ad_hc",
            "Label file must contain only AD/HC Phase 1 labels with at least one subject per class.",
            {"counts": counts},
        )


def _validate_prediction_report(check: ModalCheck, report: dict, require_real_data: bool) -> None:
    pred = report.get("subject_level_prediction", {})
    emb = report.get("embedding_output", {})
    protocol = report.get("evaluation_protocol", {})
    meta = report.get("input_metadata", {})

    prediction = pred.get("prediction")
    if prediction in {"Healthy Control", "Alzheimer's EEG Pattern"}:
        check.ok("prediction_safe_class_name", {"prediction": prediction})
    else:
        check.fail("prediction_safe_class_name", "Prediction class is not an allowed Phase 1 class.", pred)

    confidence = float(pred.get("subject_level_confidence", -1))
    if 0.0 <= confidence <= 1.0:
        check.ok("confidence_range", {"subject_level_confidence": confidence})
    else:
        check.fail("confidence_range", "Confidence must be between 0 and 1.", pred)

    if protocol.get("no_epoch_leakage") is True:
        check.ok("report_no_epoch_leakage_flag", protocol)
    else:
        check.fail("report_no_epoch_leakage_flag", "Prediction report must state no_epoch_leakage=true.", protocol)

    l2_norm = float(emb.get("l2_norm", -1))
    if emb.get("z_eeg_shape") == [256] and abs(l2_norm - 1.0) <= 0.02 and emb.get("availability_flag") == 1:
        check.ok("z_eeg_contract", emb)
    else:
        check.fail("z_eeg_contract", "z_eeg must be available, 256D, and approximately L2-normalized.", emb)

    disclaimer = str(report.get("clinical_disclaimer", ""))
    if CLINICAL_DISCLAIMER in disclaimer and "not a clinical diagnosis" in disclaimer.lower():
        check.ok("clinical_disclaimer", {"clinical_disclaimer": disclaimer})
    else:
        check.fail("clinical_disclaimer", "Clinical disclaimer is missing or unsafe.", {"value": disclaimer})

    synthetic = bool(meta.get("synthetic_data", False))
    if require_real_data and synthetic:
        check.fail("real_data_required", "Report was generated from synthetic fallback data.", meta)
    else:
        check.ok("real_data_requirement", {"require_real_data": require_real_data, "synthetic_data": synthetic})

    for key, value in report.get("visual_outputs", {}).items():
        if not value:
            check.fail(f"visual_output_{key}", "Visual output path is empty.", {"path": value})
        elif Path(value).exists():
            check.ok(f"visual_output_{key}", {"path": value})
        else:
            check.fail(f"visual_output_{key}", "Visual output file is missing.", {"path": value})


def run_modal_post_train_check(
    config_path: str = "configs/training.yaml",
    subject_id: str = "sub-001",
    fold: int | None = None,
    all_folds: bool = False,
    require_real_data: bool = True,
) -> dict:
    cfg = load_config(config_path)
    check = ModalCheck()

    active_fold = int(cfg["data"].get("active_fold", 0) if fold is None else fold)
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    _validate_labels(check, resolve_data_file(cfg, "label_file", "labels_dir"))
    split_file = resolve_data_file(cfg, "split_file", "splits_dir")
    split_exists = _check_file(check, "split_file_exists", split_file)
    _check_dir_has_files(check, "processed_epochs_exist", resolve_path(cfg["paths"]["processed_epochs_dir"]), "*_epochs.npz")
    try:
        shape_validation = validate_preprocessed_shapes(config_path)
        if shape_validation["passed"]:
            check.ok("preprocessed_shape_validation", {
                "files_checked": shape_validation["files_checked"],
                "reference_epoch_shape": shape_validation["reference_epoch_shape"],
            })
        else:
            check.fail("preprocessed_shape_validation", "Preprocessed epoch files are inconsistent.", shape_validation)
    except Exception as exc:
        check.fail("preprocessed_shape_validation", f"{type(exc).__name__}: {exc}")

    if split_exists:
        leakage = check_no_subject_leakage(str(split_file))
        if leakage.get("no_epoch_leakage") is True:
            check.ok("leakage_check_passed", leakage)
        else:
            check.fail("leakage_check_passed", "Subject leakage detected.", leakage)
    else:
        check.fail("leakage_check_passed", "Leakage check skipped because split file is missing.")

    checkpoint_dir = resolve_path(cfg["paths"]["checkpoints_dir"])
    folds_to_check = range(int(cfg["data"].get("n_splits", 5))) if all_folds else [active_fold]
    checkpoints_ok = True
    for fold_id in folds_to_check:
        checkpoints_ok = _check_file(
            check,
            f"checkpoint_fold{fold_id}_exists",
            checkpoint_dir / f"eegnet_fold{fold_id}_best.pth",
        ) and checkpoints_ok

    if checkpoints_ok:
        try:
            if all_folds:
                metrics = evaluate_all_folds(config_path, auto_train=False)
                check.ok("all_fold_evaluation_completed", {"folds_evaluated": metrics["folds_evaluated"]})
            else:
                metrics = evaluate(config_path, fold_override=active_fold, auto_train=False)
                check.ok("subject_level_evaluation_completed", {"fold": metrics["fold"], "metrics": metrics["metrics"]})
        except Exception as exc:
            check.fail("subject_level_evaluation_completed", f"{type(exc).__name__}: {exc}")

        try:
            report = predict_subject(subject_id, config_path, auto_train=False)
            check.ok("prediction_report_generated", {"subject_id": subject_id})
            _validate_prediction_report(check, report, require_real_data=require_real_data)
            embedding_file = resolve_path(cfg["paths"]["embeddings_dir"]) / f"{subject_id}_z_eeg.npy"
            _check_file(check, "subject_embedding_file_exists", embedding_file)
        except Exception as exc:
            check.fail("prediction_report_generated", f"{type(exc).__name__}: {exc}", {"subject_id": subject_id})

        try:
            from app.gradio_app import build_app

            build_app()
            check.ok("gradio_app_builds")
        except Exception as exc:
            check.fail("gradio_app_builds", f"{type(exc).__name__}: {exc}")

    summary = {
        "status": "PASS" if check.passed else "FAIL",
        "config": config_path,
        "subject_id": subject_id,
        "fold": active_fold,
        "all_folds": all_folds,
        "require_real_data": require_real_data,
        "checks": check.results,
    }
    write_json(reports_dir / "modal_post_train_check.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Modal post-training smoke test for Phase 1.")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--subject_id", default="sub-001")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback outputs. Use only for local smoke tests, not final Modal validation.",
    )
    args = parser.parse_args()

    summary = run_modal_post_train_check(
        config_path=args.config,
        subject_id=args.subject_id,
        fold=args.fold,
        all_folds=args.all_folds,
        require_real_data=not args.allow_synthetic,
    )
    print(f"Modal post-train check: {summary['status']}")
    for item in summary["checks"]:
        marker = "PASS" if item["passed"] else "FAIL"
        print(f"[{marker}] {item['name']}")
        if not item["passed"]:
            print(f"       {item['details'].get('message', item['details'])}")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
