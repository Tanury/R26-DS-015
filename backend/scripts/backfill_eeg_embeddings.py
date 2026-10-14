"""Export a z_eeg vector for every cohort subject, each from the fold that held it out.

The training run wrote full embeddings for four demo subjects only, so the cohort
projection covered 4 of 115 points and the per-subject embedding panel was empty for
everyone else. This recomputes them.

**Why not just use the deployed graph.** The served TorchScript model is one fold. Of
the 115 subjects, 92 were in that fold's *training* set, so embedding them all with it
would produce a projection whose class separation is in-sample and flattering — the
opposite of what the panel is for. Instead each subject is embedded with
`transformer_raw_fold{k}.pth` for the fold k that held it out, which is the same model
that produced its stored risk scores.

That correspondence is also the correctness check: the recomputed risk scores must
match `transformer_raw_subject_predictions.csv` for the same subject. A mismatch means
the preprocessing has drifted from the run, and the run is wrong to trust.

    python scripts/backfill_eeg_embeddings.py --run ../neuro-ai-eeg
    python scripts/verify_eeg_bundle.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.config import settings  # noqa: E402

EEG_DIR = Path(settings.eeg_model_dir)
COHORT_DIR = EEG_DIR / "cohort"

# The recomputed score must land this close to the run's stored score. Same tolerance
# as check_serving_parity.py, and for the same reason: ICA is seeded but MNE's solver
# is not bit-identical across BLAS builds.
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


def subject_embedding(model, tensor_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean-pool per-epoch outputs into one subject vector.

    `consistency` is the mean pairwise cosine between per-epoch embeddings — how much
    the encoder agrees with itself across windows of the same recording. Low values
    mean the subject vector is an average over disagreeing views and should be read
    with suspicion, so it is worth storing even though nothing gates on it.
    """
    import torch

    with torch.no_grad():
        out = model(torch.from_numpy(tensor_np).float())
        risk = out["risk_scores"].mean(dim=0).numpy()
        per_epoch = out["z_eeg"].numpy()

    pooled = per_epoch.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    pooled = pooled / norm if norm > 1e-12 else np.full_like(pooled, 1.0 / np.sqrt(pooled.size))

    unit = per_epoch / np.clip(np.linalg.norm(per_epoch, axis=1, keepdims=True), 1e-12, None)
    if unit.shape[0] > 1:
        similarity = unit @ unit.T
        upper = similarity[np.triu_indices(unit.shape[0], k=1)]
        consistency = float(np.clip(upper.mean(), 0.0, 1.0))
    else:
        consistency = 1.0
    return pooled.astype("float64"), risk, consistency


def neighbourhood_agreement(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Fraction of subjects whose nearest neighbour shares their class, in FULL 256-D.

    The scatter cannot answer this. PC1 + PC2 of a 256-D embedding capture roughly a
    fifth of the variance here, and the directions that separate classes are not the
    directions that carry the most variance — so the plot can look like noise while
    the geometry behind it is well separated. Measuring in the untruncated space is
    the only honest way to say which it is.

    Vectors are already L2-normalized, so a dot product is the cosine.
    """
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = labels[similarity.argmax(axis=1)]
    agreement = {"overall": round(float((nearest == labels).mean()), 4)}
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        agreement[str(name)] = round(float((nearest[mask] == name).mean()), 4)
    return agreement


def build_projection(vectors: dict[str, np.ndarray], reports: dict[str, dict]) -> dict:
    """PCA over every subject embedding, not just the demo four."""
    ids = sorted(vectors)
    matrix = np.stack([vectors[i] for i in ids])
    labels = np.array([reports[i]["dataset"].get("true_class", "") for i in ids])
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    variance = (s ** 2) / max(float((s ** 2).sum()), 1e-12)
    agreement = neighbourhood_agreement(matrix, labels) if len(ids) > 2 else {}
    return {
        "method": "PCA",
        "explained_variance": [round(float(variance[0]), 4), round(float(variance[1]), 4)],
        "neighbourhood_agreement": agreement,
        "note": (
            f"Subject-level z_eeg for all {len(ids)} assessed subjects, projected with PCA. "
            "Each vector comes from the cross-validation fold that held that subject out, "
            "so the separation shown is out-of-sample. These two components carry only "
            "part of the variance, so read the neighbourhood agreement above rather than "
            "the visible spread — it is measured in the full 256-D space."
        ),
        "points": [{
            "subject_id": sid,
            "x": round(float(coords[i, 0]), 5),
            "y": round(float(coords[i, 1]), 5),
            "true_class": reports[sid]["dataset"].get("true_class", ""),
            "site": reports[sid]["dataset"].get("site", ""),
        } for i, sid in enumerate(ids)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=BACKEND_DIR.parent / "neuro-ai-eeg")
    parser.add_argument("--dataset", type=Path, default=BACKEND_DIR.parents[1] / "dataset")
    parser.add_argument("--limit", type=int, default=0, help="stop after N subjects (debug)")
    parser.add_argument("--projection-only", action="store_true",
                        help="rebuild projection.json from stored vectors, no inference")
    args = parser.parse_args()

    reports = {}
    for path in COHORT_DIR.glob("*.json"):
        if path.stem in {"index", "projection", "band_reference"}:
            continue
        reports[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    # Recomputing the projection is cheap; recomputing the vectors that feed it costs
    # a full preprocessing pass over every recording. Keep them separable.
    if args.projection_only:
        stored = {}
        for path in sorted((COHORT_DIR / "embeddings").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["subject_id"] in reports:
                stored[payload["subject_id"]] = np.asarray(payload["z_eeg"], dtype="float64")
        if not stored:
            raise SystemExit("No stored embeddings; run without --projection-only first.")
        projection = build_projection(stored, reports)
        (COHORT_DIR / "projection.json").write_text(
            json.dumps(projection, indent=2), encoding="utf-8")
        print(f"projection rebuilt: {len(projection['points'])} points, "
              f"explained variance {projection['explained_variance']}")
        print(f"neighbourhood agreement: {projection['neighbourhood_agreement']}")
        return 0

    from app.services.eeg_inference_service import build_representation
    from app.services.eeg_model_loader import load_eeg_assets
    from app.services.eeg_preprocessing import preprocess
    from eeg_encoder_arch import build_encoder

    artifacts = args.run / "r26_ds015_artifacts"
    checkpoints = artifacts / "checkpoints"
    predictions_path = sorted((artifacts / "reports").glob("*_subject_predictions.csv"))
    if not predictions_path:
        raise SystemExit(f"No *_subject_predictions.csv in {artifacts / 'reports'}")
    predictions = pd.read_csv(predictions_path[-1]).set_index("subject_id")

    assets = load_eeg_assets()
    config = {**assets.preprocessing_config,
              "channel_order": list(assets.channel_order),
              "sampling_rate_hz": assets.sampling_rate_hz,
              "epoch_length_seconds": assets.epoch_length_seconds,
              "standardization": assets.standardization}
    centroids = {name: np.asarray(vec, dtype="float64")
                 for name, vec in (assets.class_centroids or {}).items()}

    print("=" * 78)
    print("BACKFILLING z_eeg FOR EVERY COHORT SUBJECT")
    print("=" * 78)
    print(f"  cohort          {len(reports)} subjects")
    print(f"  standardization {assets.standardization}")
    print(f"  tolerance       {TOLERANCE}\n")

    # One model per fold, loaded once. Five 2.3 MB encoders is nothing next to
    # reloading a checkpoint per subject.
    models: dict[int, object] = {}
    for fold in sorted(predictions["fold"].unique()):
        path = checkpoints / f"transformer_raw_fold{int(fold)}.pth"
        if not path.exists():
            raise SystemExit(f"Missing fold checkpoint: {path}")
        models[int(fold)], _ = build_encoder(path)
    print(f"  loaded {len(models)} fold checkpoints\n")

    vectors: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    mismatched: list[str] = []
    worst = 0.0

    order = sorted(reports)
    if args.limit:
        order = order[: args.limit]

    for n, subject_id in enumerate(order, start=1):
        if subject_id not in predictions.index:
            skipped.append(f"{subject_id}: absent from the predictions CSV")
            continue
        row = predictions.loc[subject_id]
        recording = find_recording(subject_id, args.dataset)
        if recording is None:
            skipped.append(f"{subject_id}: recording not found")
            print(f"  [{n:>3}/{len(order)}] {subject_id:<22} SKIP recording not found")
            continue

        try:
            prepared = preprocess(recording, config)
            # Same representation builder the API uses, so the tensor the fold model
            # sees is identical to the one serving would produce.
            tensor_np = build_representation(prepared["epochs"], assets)
            vector, risk, consistency = subject_embedding(
                models[int(row["fold"])], tensor_np)
        except Exception as exc:
            skipped.append(f"{subject_id}: {type(exc).__name__}: {exc}")
            print(f"  [{n:>3}/{len(order)}] {subject_id:<22} ERROR {type(exc).__name__}")
            continue

        # Same fold model + same preprocessing must reproduce the run's scores.
        expected = np.array([float(row[f"risk_{c}"]) for c in assets.risk_conditions])
        deviation = float(np.abs(risk - expected).max())
        worst = max(worst, deviation)
        status = "ok  " if deviation <= TOLERANCE else "DRIFT"
        if deviation > TOLERANCE:
            mismatched.append(f"{subject_id}: max|delta| {deviation:.4f}")

        vectors[subject_id] = vector
        print(f"  [{n:>3}/{len(order)}] {subject_id:<22} {status} "
              f"fold {int(row['fold'])}  delta {deviation:.4f}  consistency {consistency:.3f}")

        cosine = {name: round(float(vector @ centroid), 4)
                  for name, centroid in centroids.items()}
        nearest = max(cosine, key=cosine.get) if cosine else None

        (COHORT_DIR / "embeddings" / f"{subject_id}.json").write_text(json.dumps({
            "subject_id": subject_id,
            "dim": int(vector.size),
            "l2_norm": round(float(np.linalg.norm(vector)), 6),
            "availability_flag": 1,
            "z_eeg": [round(float(v), 6) for v in vector],
        }), encoding="utf-8")

        report = reports[subject_id]
        report["embedding"] = {
            "dim": int(vector.size),
            "l2_norm": round(float(np.linalg.norm(vector)), 6),
            "availability_flag": 1,
            "consistency": round(consistency, 4),
            "cosine_to_class_centroids": cosine,
            "nearest_centroid": nearest,
            "vector_url": f"/eeg/embeddings/{subject_id}",
        }
        # The warning named embeddings and occlusion together; only embeddings are
        # backfilled here, so it has to be narrowed rather than dropped.
        report["signal_quality"]["warnings"] = [
            ("Occlusion explainability was not exported for this subject; the training "
             "run writes it only for its demo subjects.")
            if "Embedding geometry and occlusion" in warning else warning
            for warning in report["signal_quality"].get("warnings", [])
        ]
        (COHORT_DIR / f"{subject_id}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    if vectors:
        projection = build_projection(vectors, reports)
        (COHORT_DIR / "projection.json").write_text(
            json.dumps(projection, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  embedded        {len(vectors)}/{len(order)} subjects")
    print(f"  worst deviation {worst:.4f} (tolerance {TOLERANCE})")
    if vectors:
        print(f"  projection      {len(projection['points'])} points, "
              f"explained variance {projection['explained_variance']}")
    for note in skipped:
        print(f"  SKIP  {note}")
    for note in mismatched:
        print(f"  DRIFT {note}")
    print("=" * 78)
    if mismatched:
        print("FAILED — recomputed scores do not match the run; preprocessing has drifted")
        return 1
    print("DONE — run scripts/verify_eeg_bundle.py next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
