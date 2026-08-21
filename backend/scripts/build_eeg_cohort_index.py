"""Build the EEG cohort store the backend serves.

Two modes.

`--from-workspace <dir>` is the real one: it reads the per-subject reports the
training notebook wrote to `outputs/reports/`, plus `crossval_metrics.json` and
`confound_probes.json`, and reshapes them into the API's contract.

`--fixtures` generates a development cohort when the notebook has not been run.
Subject ids, classes and recording sites are read from the actual BrainLat folder
so the fixture roster matches the real one and swapping in real reports later is a
drop-in replacement. **Scores are synthetic** and every generated report is stamped
`"fixture": true` so it can never be mistaken for a result.

    python scripts/build_eeg_cohort_index.py --fixtures --dataset ../../dataset
    python scripts/build_eeg_cohort_index.py --from-workspace ~/r26_ds015_workspace
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
EEG_DIR = BACKEND_DIR / "app" / "models" / "eeg"
COHORT_DIR = EEG_DIR / "cohort"

RISK_CONDITIONS = ["AD", "PD", "MS"]
CLASS_NAMES = ["HC", "AD", "PD", "MS"]
GROUP_DIRS = {"1_AD": "AD", "3_PD": "PD", "4_MS": "MS", "5_HC": "HC"}
RISK_BANDS = {"low_max": 0.39, "medium_max": 0.69}
BANDS = {
    "delta": [0.5, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0],
    "beta": [13.0, 30.0], "low_gamma": [30.0, 40.0],
}

# Measured on the real cohort during the dataset audit; see the implementation plan.
SEVERITY = {
    "AD": "MODERATE (age)",
    "PD": "low",
    "MS": "CRITICAL (age + site)",
    "HC": "low",
}
DISCLOSURE_STATEMENT = (
    "This model was trained on a cohort in which MS subjects are ~33 years younger "
    "than every other group and come from a single recording site. Risk scores for "
    "classes flagged CRITICAL or HIGH cannot be attributed to disease physiology on "
    "the strength of this cohort alone. Read risk_score_age_correlation before using "
    "any score: the within-negatives correlation is the decisive figure."
)
DISCLAIMER = (
    "This output is a research decision-support indicator produced by an EEG "
    "representation learning prototype. It is not a clinical diagnosis, has not been "
    "clinically validated, and must not be used as the basis of a medical decision."
)


def risk_band(score: float) -> str:
    if score <= RISK_BANDS["low_max"]:
        return "Low"
    return "Medium" if score <= RISK_BANDS["medium_max"] else "High"


# --------------------------------------------------------------------- roster
def scan_dataset(dataset_root: Path) -> list[dict[str, Any]]:
    """Real subject ids, classes and sites, straight off disk.

    Canonical ids carry class and site because six PD directory names appear under
    both AR and CL and are different people.
    """
    rows: list[dict[str, Any]] = []
    for group_dir, class_code in GROUP_DIRS.items():
        for site in ("AR", "CL"):
            site_path = dataset_root / group_dir / site
            if not site_path.is_dir():
                continue
            for subject_dir in sorted(p for p in site_path.iterdir() if p.is_dir()):
                set_files = list(subject_dir.rglob("*.set"))
                if not set_files:
                    continue
                name = set_files[0].name.lower()
                task = ("pre_epoched" if "reject" in name else "continuous")
                rows.append({
                    "subject_id": f"{class_code}-{site}-{subject_dir.name}",
                    "true_class": class_code,
                    "site": site,
                    "source_kind": task,
                })
    return rows


# ------------------------------------------------------------------- fixtures
def synth_report(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """A structurally complete report with plausible-but-invented numbers."""
    true_class = row["true_class"]

    # The true condition scores higher, but with real overlap — a fixture that is
    # perfectly separable would make the UI look better than the model ever will.
    scores: dict[str, float] = {}
    for condition in RISK_CONDITIONS:
        base = 0.62 if condition == true_class else 0.24
        scores[condition] = min(max(rng.gauss(base, 0.16), 0.02), 0.97)

    epochs_used = rng.randint(46, 80)
    clean_ratio = round(min(max(rng.gauss(0.88, 0.11), 0.42), 1.0), 4)
    grade = "Good" if clean_ratio >= 0.8 else ("Moderate" if clean_ratio >= 0.5 else "Poor")

    ica_n = rng.randint(1, 5)
    criteria_pool = [["frontal_bipolar"], ["high_frequency"], ["kurtosis"],
                     ["kurtosis", "high_frequency"]]
    ica_rejections = [
        {
            "component": i,
            "criteria": rng.choice(criteria_pool),
            "kurtosis": round(rng.uniform(0.3, 40.0), 3),
            "frontal_corr": round(rng.uniform(0.02, 0.85), 3),
            "hf_power_ratio": round(rng.uniform(0.02, 0.72), 3),
        }
        for i in range(ica_n)
    ]

    # AD-type slowing: theta up, alpha down. Keeps the band chart clinically coherent.
    slowing = 0.09 if true_class == "AD" else 0.0
    raw_bands = {
        "delta": max(rng.gauss(0.20, 0.04), 0.02),
        "theta": max(rng.gauss(0.16 + slowing, 0.04), 0.02),
        "alpha": max(rng.gauss(0.30 - slowing, 0.06), 0.02),
        "beta": max(rng.gauss(0.26, 0.05), 0.02),
        "low_gamma": max(rng.gauss(0.08, 0.02), 0.01),
    }
    total = sum(raw_bands.values())
    band_profile = {k: round(v / total, 5) for k, v in raw_bands.items()}
    band_profile["theta_alpha_ratio"] = round(
        band_profile["theta"] / band_profile["alpha"], 5)

    centroids = {
        name: round(0.72 if name == true_class else rng.uniform(0.18, 0.46), 4)
        for name in CLASS_NAMES
    }
    nearest = max(centroids, key=centroids.get)

    softmax_logits = {
        name: math.exp((2.1 if name == true_class else 0.0) + rng.gauss(0, 0.7))
        for name in CLASS_NAMES
    }
    softmax_total = sum(softmax_logits.values())
    class_probabilities = {k: round(v / softmax_total, 4) for k, v in softmax_logits.items()}

    conditions = {
        condition: {
            "risk_score": round(score, 4),
            "risk_band": risk_band(score),
            "label": f"{condition}-related EEG risk pattern",
            "epoch_score_std": round(rng.uniform(0.02, 0.11), 4),
            "epoch_score_range": [round(max(score - rng.uniform(0.05, 0.2), 0.0), 4),
                                  round(min(score + rng.uniform(0.05, 0.2), 1.0), 4)],
            "confound_severity": SEVERITY.get(condition, "unknown"),
        }
        for condition, score in scores.items()
    }

    return {
        "subject_id": row["subject_id"],
        "source": "cohort",
        "fixture": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "BrainLat (Latin American Brain Health Institute)",
            "task": "resting-state EEG",
            "site": row["site"],
            "true_class": true_class,
        },
        "risk_scores": {
            f"{c.lower()}_risk_score": round(scores[c], 4) for c in RISK_CONDITIONS
        },
        "risk_assessment": {
            "conditions": conditions,
            "highest_risk_condition": max(scores, key=scores.get),
            "scores_are_independent": True,
            "interpretation": (
                "Each score is an independent probability that the recording shows the EEG "
                "pattern associated with that condition. Scores do not sum to 1 and are not "
                "mutually exclusive: elevated scores on more than one condition are "
                "meaningful, not contradictory. These are decision-support indicators, not "
                "diagnoses."
            ),
            "risk_bands": RISK_BANDS,
        },
        "optional_four_class_prediction": {
            "predicted_class": max(class_probabilities, key=class_probabilities.get),
            "class_probabilities": class_probabilities,
        },
        "signal_quality": {
            "epochs_used": epochs_used,
            "total_epochs_generated": int(epochs_used / max(clean_ratio, 0.3)),
            "clean_epoch_ratio": clean_ratio,
            "grade": grade,
            "ica_components_removed": ica_n,
            "ica_rejections": ica_rejections,
            "channels": 128,
            "sampling_rate_hz": 256.0,
            "source_kind": row["source_kind"],
            "warnings": (
                ["Recording was stored pre-epoched; 1 s segments were reassembled with "
                 "tapered joins and are not physiologically continuous."]
                if row["source_kind"] == "pre_epoched" else []
            ),
        },
        "band_power_profile": band_profile,
        "embedding": {
            "dim": 256,
            "l2_norm": 1.0,
            "availability_flag": 1,
            "consistency": round(rng.uniform(0.72, 0.96), 4),
            "cosine_to_class_centroids": centroids,
            "nearest_centroid": nearest,
            "vector_url": f"/eeg/embeddings/{row['subject_id']}",
        },
        "explainability": {
            "scalp_region_importance": {
                region: round(rng.uniform(-0.01, 0.14), 4)
                for region in ("frontal_central", "posterior", "left_lateral", "right_lateral")
            },
            "band_importance": {
                band: round(rng.uniform(-0.01, 0.10), 4) for band in BANDS
            },
            "method": (
                "occlusion — drop in predicted-condition probability when the input is zeroed"
            ),
        },
        "confound_disclosure": {},   # merged from the model card at read time
        "model_summary": {
            "architecture": "BiLSTM + Attention",
            "input_representation": "hybrid",
            "embedding_dim": 256,
        },
        "clinical_disclaimer": DISCLAIMER,
    }


def synth_embedding(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """A unit-norm 256-D vector clustered by class, so the projection looks real."""
    anchor = rng.Random if False else None  # noqa: F841 - clarity only
    class_rng = random.Random(hash(row["true_class"]) & 0xFFFF)
    centre = [class_rng.gauss(0, 1) for _ in range(256)]
    vector = [c * 0.75 + rng.gauss(0, 0.55) for c in centre]
    vector = [abs(v) for v in vector]          # ReLU head → non-negative orthant
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    unit = [round(v / norm, 6) for v in vector]
    return {
        "subject_id": row["subject_id"],
        "dim": 256,
        "l2_norm": round(math.sqrt(sum(v * v for v in unit)), 6),
        "availability_flag": 1,
        "z_eeg": unit,
    }


def fixture_model_card(n_subjects: int, per_class: dict[str, int],
                       per_site: dict[str, int]) -> dict[str, Any]:
    return {
        "run_id": f"fixture-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": True,
        "model": {
            "architecture": "BiLSTM + Attention",
            "input_representation": "hybrid",
            "input_shape": [10, 128, 16],
            "embedding_dim": 256,
            "outputs": {
                "risk_scores": {
                    "order": RISK_CONDITIONS, "activation": "sigmoid", "independent": True,
                    "note": "Do NOT sum to 1. Elevated scores on several conditions are meaningful.",
                },
                "class_probabilities": {"order": CLASS_NAMES, "activation": "softmax"},
                "z_eeg": {"shape": [256], "normalization": "L2 unit norm after ReLU"},
            },
        },
        "training_data": {
            "dataset": "BrainLat (Latin American Brain Health Institute)",
            "n_subjects": n_subjects,
            "per_class": per_class,
            "per_site": per_site,
            "label_policy": "exclusive",
            "cross_validation": "subject-level StratifiedGroupKFold, 5 folds",
            "synthetic_run": True,
        },
        "performance": {
            "risk_macro_auc_mean": 0.78, "risk_macro_auc_std": 0.09,
            "per_condition_auc": {"AD": 0.82, "PD": 0.79, "MS": 0.74},
            "pooled_per_condition": {
                "AD": {"auc": 0.82, "auc_ci": {"ci_low": 0.71, "ci_high": 0.91}},
                "PD": {"auc": 0.79, "auc_ci": {"ci_low": 0.66, "ci_high": 0.90}},
                "MS": {"auc": 0.74, "auc_ci": {"ci_low": 0.59, "ci_high": 0.87}},
            },
            "binary_neuro_vs_hc_auc": 0.84,
            "binary_auc_ci": {"ci_low": 0.74, "ci_high": 0.93},
            "embedding_silhouette": 0.21,
        },
        "confound_disclosure": {
            "age_probe": {"mae_years": 6.05, "baseline_mae_years": 13.64,
                          "improvement_over_baseline": 0.5561, "pearson_r": 0.8977},
            "site_probe": {"balanced_accuracy": 0.5036, "majority_baseline": 0.5882},
            "risk_score_age_correlation": {
                "AD": {"pearson_r_score_vs_age": 0.575, "pearson_r_within_negatives": 0.372},
                "PD": {"pearson_r_score_vs_age": 0.497, "pearson_r_within_negatives": 0.553},
                "MS": {"pearson_r_score_vs_age": -0.555, "pearson_r_within_negatives": -0.182},
            },
            "severity_by_condition": {k: v for k, v in SEVERITY.items() if k != "HC"},
            "statement": DISCLOSURE_STATEMENT,
        },
        "intended_use": {
            "purpose": (
                "Research decision-support. Produces a 256-D z_eeg embedding for multimodal "
                "fusion and three independent EEG risk-pattern scores."
            ),
            "out_of_scope": [
                "Clinical diagnosis or any medical decision",
                "Populations unlike the training cohort (Latin American, two sites, n~118)",
                "EEG montages other than BioSemi 128 A1-D32, or non-resting-state protocols",
            ],
            "disclaimer": DISCLAIMER,
        },
    }


# ------------------------------------------------------- real notebook reports
def from_workspace(workspace: Path) -> tuple[list[dict], dict, list[dict]]:
    """Reshape notebook `outputs/reports/*_report.json` into the API contract."""
    reports_dir = workspace / "outputs" / "reports"
    if not reports_dir.is_dir():
        raise SystemExit(f"No reports directory at {reports_dir}")

    card_path = workspace / "exported_model" / "model_card.json"
    if not card_path.exists():
        raise SystemExit(
            f"No model card at {card_path}. Run Section 24 of the Colab notebook first."
        )
    card = json.loads(card_path.read_text(encoding="utf-8"))

    reports, embeddings = [], []
    for path in sorted(reports_dir.glob("*_report.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        subject_id = raw["input_metadata"]["subject_id"]
        stage3 = raw["preprocessing_summary"]["stage_3_epoching"]
        stage2 = raw["preprocessing_summary"]["stage_2_ica"]
        assessment = raw["risk_assessment"]

        reports.append({
            "subject_id": subject_id,
            "source": "cohort",
            "generated_at": raw.get("generated_at", ""),
            "dataset": {
                "name": raw["dataset"]["name"],
                "task": raw["dataset"].get("task", "resting-state EEG"),
                "site": raw["input_metadata"].get("site", ""),
                "true_class": raw["input_metadata"].get("class", ""),
            },
            "risk_scores": raw["risk_scores"],
            "risk_assessment": {
                "conditions": assessment["conditions"],
                "highest_risk_condition": assessment["highest_risk_condition"],
                "scores_are_independent": True,
                "interpretation": assessment["interpretation"],
                "risk_bands": assessment.get("risk_bands", RISK_BANDS),
            },
            "optional_four_class_prediction": {
                "predicted_class": raw["optional_four_class_prediction"]["predicted_class"],
                "class_probabilities":
                    raw["optional_four_class_prediction"]["class_probabilities"],
            },
            "signal_quality": {
                "epochs_used": stage3["epochs_after_cap"],
                "total_epochs_generated": stage3["total_epochs_generated"],
                "clean_epoch_ratio": stage3["clean_epoch_ratio"],
                "grade": stage3["signal_quality"],
                "ica_components_removed": len(stage2.get("excluded_components", [])),
                "ica_rejections": [
                    {
                        "component": c["component"], "criteria": c["criteria_fired"],
                        "kurtosis": c.get("kurtosis"), "frontal_corr": c.get("frontal_corr"),
                        "hf_power_ratio": c.get("hf_power_ratio"),
                    }
                    for c in stage2.get("component_scores", []) if c.get("excluded")
                ],
                "channels": raw["input_metadata"]["n_channels"],
                "sampling_rate_hz": raw["input_metadata"].get("resampled_to_hz", 256.0),
                "source_kind": raw["input_metadata"].get("source_kind", "continuous"),
                "warnings": raw.get("quality_control", {}).get("warnings", []),
            },
            "band_power_profile": {
                k: v for k, v in raw.get("band_power_profile", {}).items() if "__" not in k
            },
            "embedding": {
                "dim": raw["embedding_output"]["embedding_dim"],
                "l2_norm": raw["embedding_output"]["l2_norm"],
                "availability_flag": raw["embedding_output"]["availability_flag"],
                "consistency": raw["embedding_output"]["embedding_consistency"],
                "cosine_to_class_centroids":
                    raw["embedding_output"]["cosine_to_class_centroids"],
                "nearest_centroid": raw["embedding_output"]["nearest_centroid"],
                "vector_url": f"/eeg/embeddings/{subject_id}",
            },
            "explainability": {
                "scalp_region_importance":
                    raw.get("explainability", {}).get("scalp_region_importance", {}),
                "band_importance": raw.get("explainability", {}).get("band_importance", {}),
                "method": raw.get("explainability", {}).get("method", ""),
            },
            "confound_disclosure": {},
            "model_summary": {
                "architecture": raw["model_summary"]["architecture"],
                "input_representation": raw["model_summary"]["input_representation"],
                "embedding_dim": raw["model_summary"]["embedding_dim"],
            },
            "clinical_disclaimer": raw["clinical_disclaimer"],
        })

        vector = raw["embedding_output"].get("z_eeg")
        if isinstance(vector, list) and len(vector) > 4:
            embeddings.append({
                "subject_id": subject_id, "dim": len(vector),
                "l2_norm": raw["embedding_output"]["l2_norm"],
                "availability_flag": 1, "z_eeg": vector,
            })

    if not reports:
        raise SystemExit(f"No *_report.json files found in {reports_dir}")
    return reports, card, embeddings


# ---------------------------------------------------------------------- write
def index_row(report: dict[str, Any]) -> dict[str, Any]:
    scores = {
        c: report["risk_assessment"]["conditions"][c]["risk_score"]
        for c in report["risk_assessment"]["conditions"]
    }
    top = report["risk_assessment"]["highest_risk_condition"]
    return {
        "subject_id": report["subject_id"],
        "true_class": report["dataset"].get("true_class", ""),
        "site": report["dataset"].get("site", ""),
        "source_kind": report["signal_quality"]["source_kind"],
        "signal_quality": report["signal_quality"]["grade"],
        "epochs_used": report["signal_quality"]["epochs_used"],
        "age": report.get("age"),
        "highest_risk_condition": top,
        "highest_risk_score": scores.get(top, 0.0),
        "risk_scores": scores,
        "confound_severity": SEVERITY.get(top, "unknown"),
    }


def build_projection(embeddings: list[dict], reports: list[dict]) -> dict[str, Any]:
    """2-D PCA over subject embeddings, computed here so the API stays cheap."""
    import numpy as np

    by_id = {r["subject_id"]: r for r in reports}
    matrix = np.array([e["z_eeg"] for e in embeddings], dtype="float64")
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    variance = (s ** 2) / max((s ** 2).sum(), 1e-12)

    return {
        "method": "PCA",
        "explained_variance": [round(float(variance[0]), 4), round(float(variance[1]), 4)],
        "note": (
            "Subject-level z_eeg projected with PCA. Embeddings come from one fold's "
            "model, so this is a single coordinate system."
        ),
        "points": [
            {
                "subject_id": e["subject_id"],
                "x": round(float(coords[i, 0]), 5),
                "y": round(float(coords[i, 1]), 5),
                "true_class": by_id.get(e["subject_id"], {}).get("dataset", {}).get("true_class", ""),
                "site": by_id.get(e["subject_id"], {}).get("dataset", {}).get("site", ""),
            }
            for i, e in enumerate(embeddings)
        ],
    }


def write_store(reports: list[dict], card: dict, embeddings: list[dict]) -> None:
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    (COHORT_DIR / "embeddings").mkdir(exist_ok=True)

    for report in reports:
        (COHORT_DIR / f"{report['subject_id']}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    (COHORT_DIR / "index.json").write_text(
        json.dumps([index_row(r) for r in reports], indent=2), encoding="utf-8")

    for embedding in embeddings:
        (COHORT_DIR / "embeddings" / f"{embedding['subject_id']}.json").write_text(
            json.dumps(embedding), encoding="utf-8")

    if embeddings:
        (COHORT_DIR / "projection.json").write_text(
            json.dumps(build_projection(embeddings, reports), indent=2), encoding="utf-8")

    (EEG_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-workspace", type=Path, default=None,
                        help="Notebook workspace containing outputs/reports/")
    parser.add_argument("--fixtures", action="store_true",
                        help="Generate a development cohort (synthetic scores)")
    parser.add_argument("--dataset", type=Path,
                        default=BACKEND_DIR.parents[1] / "dataset",
                        help="BrainLat folder, used for the fixture roster")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.from_workspace:
        reports, card, embeddings = from_workspace(args.from_workspace)
        print(f"Read {len(reports)} reports from {args.from_workspace}")
    elif args.fixtures:
        rng = random.Random(args.seed)
        roster = scan_dataset(args.dataset)
        if not roster:
            raise SystemExit(
                f"No subjects found under {args.dataset}. Pass --dataset explicitly."
            )
        reports = [synth_report(row, rng) for row in roster]
        embeddings = [synth_embedding(row, rng) for row in roster]
        per_class: dict[str, int] = {}
        per_site: dict[str, int] = {}
        for row in roster:
            per_class[row["true_class"]] = per_class.get(row["true_class"], 0) + 1
            per_site[row["site"]] = per_site.get(row["site"], 0) + 1
        card = fixture_model_card(len(roster), per_class, per_site)
        print(f"Generated {len(reports)} FIXTURE reports from the roster in {args.dataset}")
        print(f"  per class: {per_class}")
        print(f"  per site : {per_site}")
        print("  NOTE: scores are synthetic. Every report carries \"fixture\": true.")
    else:
        raise SystemExit("Pass either --from-workspace <dir> or --fixtures.")

    write_store(reports, card, embeddings)
    print(f"\nWrote cohort store to {COHORT_DIR}")
    print(f"  index.json         {len(reports)} subjects")
    print(f"  embeddings/        {len(embeddings)} vectors")
    print(f"  model_card.json    run_id={card.get('run_id')}")


if __name__ == "__main__":
    main()
