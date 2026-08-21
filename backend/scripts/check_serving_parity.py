"""Prove the served model reproduces the training run, on real recordings.

Runs real `.set` files through the serving preprocessing and the deployed
TorchScript graph, then compares against the risk scores the training run recorded
for the **same subjects under the same fold**. Any drift means serving and training
disagree — a failure that is otherwise invisible, because the tensors keep their
shape and the model keeps returning confident numbers.

Only subjects held out by the exported fold are valid comparisons: every other row
in the predictions CSV was scored by a different fold's weights.

    python scripts/check_serving_parity.py
    python scripts/check_serving_parity.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402

from app.core.config import settings  # noqa: E402

TOLERANCE = 0.05


def find_recording(subject_id: str, dataset_root: Path) -> Path | None:
    """Map a canonical id such as AD-AR-sub-30001 back onto its .set file."""
    class_code, site, name = subject_id.split("-", 2)
    folder = {"AD": "1_AD", "PD": "3_PD", "MS": "4_MS", "HC": "5_HC"}.get(class_code)
    if folder is None:
        return None
    subject_dir = dataset_root / folder / site / name
    if not subject_dir.is_dir():
        return None
    candidates = sorted(subject_dir.rglob("*.set"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=BACKEND_DIR.parent / "neuro-ai-eeg")
    parser.add_argument("--dataset", type=Path, default=BACKEND_DIR.parents[1] / "dataset")
    parser.add_argument("--limit", type=int, default=6,
                        help="Subjects to check (each costs ~10 s of preprocessing)")
    args = parser.parse_args()

    from app.services.eeg_inference_service import infer
    from app.services.eeg_model_loader import load_eeg_assets
    from app.services.eeg_preprocessing import preprocess

    card = json.loads((Path(settings.eeg_model_dir) / "model_card.json")
                      .read_text(encoding="utf-8"))
    exported_fold = int(card["model"]["exported_fold"])

    reports_dir = args.run / "r26_ds015_artifacts" / "reports"
    candidates = sorted(reports_dir.glob("*_subject_predictions.csv"))
    if not candidates:
        raise SystemExit(f"No *_subject_predictions.csv in {reports_dir}")
    predictions = pd.read_csv(candidates[-1])
    held_out = predictions[predictions["fold"] == exported_fold]

    assets = load_eeg_assets()
    config = {**assets.preprocessing_config,
              "channel_order": list(assets.channel_order),
              "sampling_rate_hz": assets.sampling_rate_hz,
              "epoch_length_seconds": assets.epoch_length_seconds,
              "standardization": assets.standardization}

    print("=" * 78)
    print("SERVING PARITY — deployed model vs the training run")
    print("=" * 78)
    print(f"  exported fold   {exported_fold} ({len(held_out)} held-out subjects)")
    print(f"  representation  {assets.representation} {assets.input_shape}")
    print(f"  standardization {assets.standardization}")
    print(f"  tolerance       {TOLERANCE}\n")

    # Spread the sample across classes rather than taking the first N.
    sample = held_out.groupby("true_class", group_keys=False).head(
        max(1, args.limit // held_out["true_class"].nunique()))
    sample = sample.head(args.limit)

    checked, worst, failures = 0, 0.0, []
    for _i, row in sample.iterrows():
        subject_id = str(row["subject_id"])
        path = find_recording(subject_id, args.dataset)
        if path is None:
            print(f"  {subject_id:<22} SKIP (recording not found)")
            continue
        try:
            report = infer(preprocess(path, config), subject_id=subject_id, explain=False)
        except Exception as exc:
            failures.append(f"{subject_id}: {type(exc).__name__}: {exc}")
            print(f"  {subject_id:<22} ERROR {type(exc).__name__}: {exc}")
            continue

        deltas = {
            condition: abs(report.risk_scores[f"{condition.lower()}_risk_score"]
                           - float(row[f"risk_{condition}"]))
            for condition in assets.risk_conditions
        }
        subject_worst = max(deltas.values())
        worst = max(worst, subject_worst)
        checked += 1
        status = "ok  " if subject_worst <= TOLERANCE else "DRIFT"
        print(f"  {subject_id:<22} {status} true={row['true_class']:<3} "
              f"max|delta|={subject_worst:.4f}  "
              + " ".join(f"{c}={report.risk_scores[f'{c.lower()}_risk_score']:.3f}"
                         f"/{float(row[f'risk_{c}']):.3f}"
                         for c in assets.risk_conditions))
        if subject_worst > TOLERANCE:
            failures.append(f"{subject_id}: max|delta| {subject_worst:.4f}")

    print("\n" + "=" * 78)
    print(f"checked {checked} subject(s), worst deviation {worst:.4f}")
    if failures:
        print(f"PARITY FAILED — {len(failures)} subject(s) drifted")
        for line in failures:
            print(f"  {line}")
        print("\nServing preprocessing no longer matches training. The usual cause is a")
        print("change to standardization, filtering or epoching applied after the model")
        print("was trained. Check the bundle's `standardization` field first.")
        return 1
    print("PARITY OK — serving reproduces the training run")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
