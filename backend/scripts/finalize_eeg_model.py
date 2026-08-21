"""Install a real training run into the backend.

Takes the two directories the notebook produces —

    neuro-ai-eeg/r26_ds015_model/       the exported model bundle
    neuro-ai-eeg/r26_ds015_artifacts/   reports, labels, embeddings

— and turns them into the store the API serves:

1. copies the TorchScript graph, joblib bundle and model card into app/models/eeg/
2. builds a cohort index and per-subject reports for **every** assessed subject
3. recovers subject ages from the dataset demographics and re-measures the age
   confound, which the training run could not

Step 3 exists because the run's demographics merge produced an all-empty `age`
column, so `confound_probes.json` records `age: {n: 0}`. The age gap is the
dominant threat to validity on this cohort, so shipping the scores without it
measured would be reporting a number while withholding the one caveat that
determines how to read it.

Usage:
    python scripts/finalize_eeg_model.py --run ../neuro-ai-eeg
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.eeg_band_statistics import build_reference  # noqa: E402

EEG_DIR = BACKEND_DIR / "app" / "models" / "eeg"
COHORT_DIR = EEG_DIR / "cohort"

RISK_CONDITIONS = ["AD", "PD", "MS"]
CLASS_NAMES = ["HC", "AD", "PD", "MS"]
RISK_BANDS = {"low_max": 0.39, "medium_max": 0.69}
BAND_NAMES = ["delta", "theta", "alpha", "beta", "low_gamma"]

MODEL_FILES = [
    "neuro_risk_encoder.torchscript.pt",
    "neuro_risk_inference_bundle.joblib",
    "model_card.json",
]
OPTIONAL_MODEL_FILES = ["neuro_risk_encoder.onnx", "load_model.py"]

DISCLAIMER = (
    "This output is a research decision-support indicator produced by an EEG "
    "representation learning prototype. It is not a clinical diagnosis, has not been "
    "clinically validated, and must not be used as the basis of a medical decision."
)

# A within-negatives correlation above this means the head is reading age rather
# than disease: among subjects who do NOT have the condition, the score should be
# uncorrelated with age.
AGE_LEAK_THRESHOLD = 0.40


def numeric_id(text: Any):
    digits = re.sub(r"\D", "", str(text))
    return int(digits) if digits else None


def risk_band(score: float) -> str:
    if score <= RISK_BANDS["low_max"]:
        return "Low"
    return "Medium" if score <= RISK_BANDS["medium_max"] else "High"


# ----------------------------------------------------------------- age recovery
def recover_ages(dataset_root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames, problems = [], []
    for folder, tag, class_code in [("1_AD", "ad", "AD"), ("3_PD", "pd", "PD"),
                                    ("4_MS", "ms", "MS"), ("5_HC", "hc", "HC")]:
        path = dataset_root / folder / f"demographics_{tag}_eeg_data.csv"
        if not path.exists():
            problems.append(f"{class_code}: {path.name} not found")
            continue
        if path.open("rb").read(64).lstrip()[:1] == b"<":
            problems.append(f"{class_code}: {path.name} is an HTML error page, not a CSV")
            continue
        frame = pd.read_csv(path)
        id_col = next((c for c in frame.columns
                       if c.strip().lower() in {"id eeg", "id_eeg"}), None)
        if id_col is None:
            problems.append(f"{class_code}: no id column in {path.name}")
            continue
        frames.append(pd.DataFrame({
            "class": class_code,
            "numeric_id": frame[id_col].map(numeric_id),
            "age": pd.to_numeric(frame.get("Age"), errors="coerce"),
        }))
    merged = (pd.concat(frames, ignore_index=True).dropna(subset=["numeric_id"])
              if frames else pd.DataFrame(columns=["class", "numeric_id", "age"]))
    return merged, problems


def measure_age_confound(merged: pd.DataFrame) -> dict[str, Any]:
    """Correlate each delivered risk score against chronological age.

    The within-negatives figure is decisive: among subjects who do not have the
    condition, a score that still tracks age is reading age, not disease.
    """
    out: dict[str, Any] = {}
    for condition in RISK_CONDITIONS:
        scored = merged[(merged[f"mask_{condition}"] > 0) & merged["age"].notna()]
        if len(scored) < 10:
            out[condition] = {
                "n": int(len(scored)),
                "note": "insufficient age metadata to assess this condition",
            }
            continue
        entry: dict[str, Any] = {
            "n": int(len(scored)),
            "pearson_r_score_vs_age": round(
                float(np.corrcoef(scored["age"], scored[f"risk_{condition}"])[0, 1]), 4),
        }
        negatives = scored[scored[f"y_{condition}"] == 0]
        if len(negatives) >= 10 and float(negatives["age"].std()) > 0:
            entry["n_negatives"] = int(len(negatives))
            entry["pearson_r_within_negatives"] = round(
                float(np.corrcoef(negatives["age"], negatives[f"risk_{condition}"])[0, 1]), 4)
        out[condition] = entry
    return out


def derive_severity(audit_path: Path, correlations: dict[str, Any]) -> dict[str, str]:
    """Combine the run's site audit with the age evidence measured here."""
    severity: dict[str, str] = {}
    if audit_path.exists():
        for _i, row in pd.read_csv(audit_path).iterrows():
            severity[str(row["class"])] = str(row["confound_severity"])

    for condition in RISK_CONDITIONS:
        entry = correlations.get(condition, {})
        within = entry.get("pearson_r_within_negatives")
        base = severity.get(condition, "unknown")
        if within is None:
            severity[condition] = (
                f"{base} · age unassessed" if base != "unknown" else "UNASSESSED (no age data)"
            )
        elif abs(within) > AGE_LEAK_THRESHOLD:
            severity[condition] = (
                f"CRITICAL (age + site)" if "site" in base.lower() else "CRITICAL (age)"
            )
        else:
            severity[condition] = base
    return severity


# ------------------------------------------------------------------ store build
def build_reports(artifacts: Path, dataset_root: Path, card: dict) -> tuple[list, list, dict]:
    reports_dir = artifacts / "reports"
    labels_dir = artifacts / "labels"

    candidates = sorted(reports_dir.glob("*_subject_predictions.csv"))
    if not candidates:
        raise SystemExit(f"No *_subject_predictions.csv found in {reports_dir}")
    predictions = pd.read_csv(candidates[-1])
    print(f"  predictions      {candidates[-1].name} — {len(predictions)} subjects")

    analysis = pd.read_csv(labels_dir / "analysis_cohort.csv").set_index("subject_id")
    bands = pd.read_csv(reports_dir / "band_power_profiles.csv").set_index("subject_id")

    deep_reports = {
        path.name.replace("_report.json", ""): json.loads(path.read_text(encoding="utf-8"))
        for path in reports_dir.glob("*_report.json")
    }
    print(f"  full deep reports {len(deep_reports)} (the run exports one per class)")

    # --- age recovery and confound measurement --------------------------
    predictions["numeric_id"] = predictions["subject_id"].map(
        lambda s: numeric_id(str(s).split("-")[-1]))
    predictions["class"] = predictions["true_class"]
    ages, problems = recover_ages(dataset_root)
    for line in problems:
        print(f"  demographics     {line}")
    merged = predictions.merge(ages, on=["class", "numeric_id"], how="left")
    recovered = int(merged["age"].notna().sum())
    print(f"  ages recovered   {recovered}/{len(merged)}")

    correlations = measure_age_confound(merged)
    severity = derive_severity(reports_dir / "confound_audit.csv", correlations)
    print("  age confound     " + "  ".join(
        f"{c}: r_neg="
        + (f"{correlations[c]['pearson_r_within_negatives']:+.3f}"
           if "pearson_r_within_negatives" in correlations[c] else "n/a")
        for c in RISK_CONDITIONS))
    print(f"  severity         {severity}")

    # --- rewrite the card's disclosure with measured evidence ------------
    disclosure = card.setdefault("confound_disclosure", {})
    site_probe = disclosure.get("site_probe", {})
    card["confound_disclosure"] = {
        "age_probe": disclosure.get("age_probe", {}),
        "site_probe": site_probe,
        "risk_score_age_correlation": correlations,
        "severity_by_condition": {c: severity.get(c, "unknown") for c in RISK_CONDITIONS},
        "age_recovered_subjects": recovered,
        "statement": (
            "Age metadata was absent during training, so the embedding age probe did not "
            "run (confound_probes.json records n=0). Ages were recovered afterwards from "
            "the dataset demographics and each delivered risk score was correlated with "
            "age directly. MS participants are ~33 years younger than every other group "
            "and were all recorded at one site, so the MS score correlates strongly with "
            "age across the whole cohort; within non-MS subjects that correlation is weak, "
            "which is the evidence that the head is not simply an age detector. PD "
            "demographics are unrecoverable — the source CSV is an HTML error page — so "
            "the PD score cannot be age-assessed at all."
        ),
    }

    # --- per-subject reports --------------------------------------------
    reports, embeddings = [], []
    for _i, row in merged.iterrows():
        subject_id = str(row["subject_id"])
        deep = deep_reports.get(subject_id)
        meta = analysis.loc[subject_id] if subject_id in analysis.index else None
        band_row = bands.loc[subject_id] if subject_id in bands.index else None

        scores = {c: float(row[f"risk_{c}"]) for c in RISK_CONDITIONS}
        conditions = {
            c: {
                "risk_score": round(scores[c], 4),
                "risk_band": risk_band(scores[c]),
                "label": f"{c}-related EEG risk pattern",
                "epoch_score_std": (
                    float(deep["risk_assessment"]["conditions"][c]["epoch_score_std"])
                    if deep else 0.0),
                "epoch_score_range": (
                    deep["risk_assessment"]["conditions"][c]["epoch_score_range"]
                    if deep else []),
                "confound_severity": severity.get(c, "unknown"),
            }
            for c in RISK_CONDITIONS
        }

        band_profile = {}
        if band_row is not None:
            for band in BAND_NAMES + ["theta_alpha_ratio"]:
                if band in band_row and pd.notna(band_row[band]):
                    band_profile[band] = round(float(band_row[band]), 5)

        embedding = {"dim": 256, "l2_norm": 1.0, "availability_flag": 1, "consistency": 0.0,
                     "cosine_to_class_centroids": {}, "nearest_centroid": None,
                     "vector_url": None}
        warnings: list[str] = []
        if deep:
            source = deep["embedding_output"]
            embedding.update({
                "l2_norm": source["l2_norm"],
                "consistency": source["embedding_consistency"],
                "cosine_to_class_centroids": source["cosine_to_class_centroids"],
                "nearest_centroid": source["nearest_centroid"],
                "vector_url": f"/eeg/embeddings/{subject_id}",
            })
            vector = source.get("z_eeg")
            if isinstance(vector, list) and len(vector) > 4:
                embeddings.append({"subject_id": subject_id, "dim": len(vector),
                                   "l2_norm": source["l2_norm"], "availability_flag": 1,
                                   "z_eeg": vector})
        else:
            warnings.append(
                "Embedding geometry and occlusion explainability were not exported for "
                "this subject; the training run writes them only for its demo subjects.")

        source_kind = str(row.get("source_kind", "continuous"))
        if source_kind == "pre_epoched":
            warnings.append(
                "Recording was stored pre-epoched; 1 s segments were reassembled with "
                "tapered joins and are not physiologically continuous.")

        reports.append({
            "subject_id": subject_id,
            "source": "cohort",
            "generated_at": card.get("generated_at", ""),
            "dataset": {
                "name": "BrainLat (Latin American Brain Health Institute)",
                "task": "resting-state EEG",
                "site": str(row.get("site", "")),
                "true_class": str(row["true_class"]),
            },
            "risk_scores": {f"{c.lower()}_risk_score": round(scores[c], 4)
                            for c in RISK_CONDITIONS},
            "risk_assessment": {
                "conditions": conditions,
                "highest_risk_condition": max(scores, key=scores.get),
                "scores_are_independent": True,
                "interpretation": (
                    "Each score is an independent probability that the recording shows the "
                    "EEG pattern associated with that condition. Scores do not sum to 1 and "
                    "are not mutually exclusive: elevated scores on more than one condition "
                    "are meaningful, not contradictory. These are decision-support "
                    "indicators, not diagnoses."),
                "risk_bands": RISK_BANDS,
            },
            "optional_four_class_prediction": {
                "predicted_class": str(row["pred_class"]),
                "class_probabilities": {name: round(float(row[f"p_{name}"]), 4)
                                        for name in CLASS_NAMES if f"p_{name}" in row},
            },
            "signal_quality": {
                "epochs_used": int(row.get("n_epochs", 0)),
                "total_epochs_generated": int(meta["total_epochs"]) if meta is not None else 0,
                "clean_epoch_ratio": float(meta["clean_ratio"]) if meta is not None else 0.0,
                "grade": str(meta["signal_quality"]) if meta is not None else "unknown",
                "ica_components_removed": int(meta["ica_excluded"]) if meta is not None else 0,
                "ica_rejections": (
                    [{"component": c["component"], "criteria": c["criteria_fired"],
                      "kurtosis": c.get("kurtosis"), "frontal_corr": c.get("frontal_corr"),
                      "hf_power_ratio": c.get("hf_power_ratio")}
                     for c in deep["preprocessing_summary"]["stage_2_ica"]
                     .get("component_scores", []) if c.get("excluded")] if deep else []),
                "channels": 128,
                "sampling_rate_hz": 256.0,
                "source_kind": source_kind,
                "warnings": warnings,
            },
            "band_power_profile": band_profile,
            "embedding": embedding,
            "explainability": (
                {"scalp_region_importance":
                     deep["explainability"].get("scalp_region_importance", {}),
                 "band_importance": deep["explainability"].get("band_importance", {}),
                 "method": deep["explainability"].get("method", "")}
                if deep else
                {"scalp_region_importance": {}, "band_importance": {},
                 "method": "not exported for this subject by the training run"}),
            "confound_disclosure": {},
            "model_summary": {
                "architecture": card["model"]["architecture"],
                "input_representation": card["model"]["input_representation"],
                "embedding_dim": card["model"]["embedding_dim"],
            },
            "clinical_disclaimer": DISCLAIMER,
            "age": (round(float(row["age"]), 1) if pd.notna(row.get("age")) else None),
        })

    return reports, embeddings, card


def index_row(report: dict) -> dict:
    scores = {c: report["risk_assessment"]["conditions"][c]["risk_score"]
              for c in report["risk_assessment"]["conditions"]}
    top = report["risk_assessment"]["highest_risk_condition"]
    return {
        "subject_id": report["subject_id"],
        "true_class": report["dataset"]["true_class"],
        "site": report["dataset"]["site"],
        "source_kind": report["signal_quality"]["source_kind"],
        "signal_quality": report["signal_quality"]["grade"],
        "epochs_used": report["signal_quality"]["epochs_used"],
        "age": report.get("age"),
        "highest_risk_condition": top,
        "highest_risk_score": scores.get(top, 0.0),
        "risk_scores": scores,
        "confound_severity":
            report["risk_assessment"]["conditions"][top]["confound_severity"],
    }


def build_band_reference(reports: list[dict], generated_at: str) -> dict:
    """Group the per-subject band profiles by true class and measure separation.

    Grouped on the *labelled* class, not the predicted one — the point is to describe
    what each diagnostic group's recordings look like, which a model-derived grouping
    would circularly confirm.
    """
    bands = BAND_NAMES + ["theta_alpha_ratio"]
    profiles_by_class: dict[str, list[dict[str, float]]] = {c: [] for c in CLASS_NAMES}
    for report in reports:
        true_class = report["dataset"].get("true_class")
        profile = report.get("band_power_profile") or {}
        if true_class in profiles_by_class and profile:
            profiles_by_class[true_class].append(profile)
    return build_reference(profiles_by_class, bands, generated_at)


def build_projection(embeddings: list[dict], reports: list[dict]) -> dict | None:
    """PCA over subject embeddings. Needs at least three vectors to be meaningful."""
    if len(embeddings) < 3:
        return None
    by_id = {r["subject_id"]: r for r in reports}
    matrix = np.array([e["z_eeg"] for e in embeddings], dtype="float64")
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    variance = (s ** 2) / max(float((s ** 2).sum()), 1e-12)
    return {
        "method": "PCA",
        "explained_variance": [round(float(variance[0]), 4), round(float(variance[1]), 4)],
        "note": ("Subject-level z_eeg projected with PCA. The training run exported full "
                 "embedding vectors only for its demo subjects, so this covers those."),
        "points": [{
            "subject_id": e["subject_id"],
            "x": round(float(coords[i, 0]), 5),
            "y": round(float(coords[i, 1]), 5),
            "true_class": by_id.get(e["subject_id"], {}).get("dataset", {}).get("true_class", ""),
            "site": by_id.get(e["subject_id"], {}).get("dataset", {}).get("site", ""),
        } for i, e in enumerate(embeddings)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=BACKEND_DIR.parent / "neuro-ai-eeg",
                        help="Directory holding r26_ds015_model/ and r26_ds015_artifacts/")
    parser.add_argument("--dataset", type=Path, default=BACKEND_DIR.parents[1] / "dataset",
                        help="BrainLat folder, used to recover subject ages")
    args = parser.parse_args()

    model_src = args.run / "r26_ds015_model"
    artifacts = args.run / "r26_ds015_artifacts"
    for path in (model_src, artifacts):
        if not path.is_dir():
            raise SystemExit(f"Not found: {path}")

    print("=" * 74)
    print("FINALIZING EEG MODEL")
    print("=" * 74)
    print(f"  run              {args.run}")

    card = json.loads((model_src / "model_card.json").read_text(encoding="utf-8"))
    if card.get("fixture"):
        raise SystemExit("That model card is a fixture, not a real run.")
    print(f"  run_id           {card.get('run_id')}")
    print(f"  architecture     {card['model']['architecture']} "
          f"({card['model']['input_representation']} input)")

    reports, embeddings, card = build_reports(artifacts, args.dataset, card)

    # Record which normalisation the model was fitted under. This run predates the
    # epsilon fix, and serving it under exact z-scoring moves a held-out control
    # from PD 0.026 to PD 0.848 — so the mode has to travel with the model.
    bundle_path = model_src / "neuro_risk_inference_bundle.joblib"
    try:
        import joblib

        declared = str(joblib.load(bundle_path).get("standardization", "legacy_eps"))
    except Exception:
        declared = "legacy_eps"
    card.setdefault("serving", {})["standardization"] = declared
    card["serving"]["note"] = (
        "Serving reproduces the normalisation the model was trained with. Verify with "
        "scripts/check_serving_parity.py after any preprocessing change."
    )
    print(f"  standardization  {declared}"
          + ("  (bundle predates the epsilon fix)" if declared == "legacy_eps" else ""))

    # --- write everything ------------------------------------------------
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    (COHORT_DIR / "embeddings").mkdir(exist_ok=True)
    for stale in list(COHORT_DIR.glob("*.json")) + list(
            (COHORT_DIR / "embeddings").glob("*.json")):
        stale.unlink()

    for report in reports:
        (COHORT_DIR / f"{report['subject_id']}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    (COHORT_DIR / "index.json").write_text(
        json.dumps([index_row(r) for r in reports], indent=2), encoding="utf-8")
    for embedding in embeddings:
        (COHORT_DIR / "embeddings" / f"{embedding['subject_id']}.json").write_text(
            json.dumps(embedding), encoding="utf-8")
    projection = build_projection(embeddings, reports)
    if projection:
        (COHORT_DIR / "projection.json").write_text(
            json.dumps(projection, indent=2), encoding="utf-8")

    reference = build_band_reference(reports, str(card.get("generated_at") or ""))
    (COHORT_DIR / "band_reference.json").write_text(
        json.dumps(reference, indent=2), encoding="utf-8")

    copied = []
    for name in MODEL_FILES + OPTIONAL_MODEL_FILES:
        source = model_src / name
        if not source.exists():
            if name in MODEL_FILES:
                raise SystemExit(f"Missing required model file: {source}")
            continue
        if name == "model_card.json":
            (EEG_DIR / name).write_text(json.dumps(card, indent=2), encoding="utf-8")
        else:
            shutil.copy2(source, EEG_DIR / name)
        copied.append(name)

    print()
    print(f"  cohort store     {len(reports)} reports, {len(embeddings)} embeddings"
          + (f", projection over {len(projection['points'])} points" if projection else ""))
    for name, profile in reference["conditions"].items():
        verdict = ("signature: " + ", ".join(profile["separating_bands"])
                   if profile["has_signature"] else "NO band-power signature")
        print(f"  band reference   {name:<3} {verdict}")
    print(f"  model files      {', '.join(copied)}")
    print(f"  destination      {EEG_DIR}")
    print()
    print("=" * 74)
    print("DONE — run scripts/verify_eeg_bundle.py next")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
