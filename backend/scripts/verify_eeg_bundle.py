"""Contract check for the EEG bundle the API serves.

Run after swapping in a new model or cohort store. Catches the failure modes that
are otherwise silent: a cohort built from a different run than the deployed model,
risk scores that have quietly become a softmax, or a report that no longer matches
the response schema.

    python scripts/verify_eeg_bundle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.services import eeg_cohort_service as cohort  # noqa: E402


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    print("=" * 72)
    print("EEG BUNDLE VERIFICATION")
    print("=" * 72)
    print(f"model dir: {settings.eeg_model_dir}\n")

    # --- model card ------------------------------------------------------
    try:
        card = cohort.load_model_card()
        print(f"  model card      OK   run_id={card.run_id}")
        print(f"                       {card.architecture} · {card.input_representation} "
              f"· {card.embedding_dim}-D")
        print(f"                       inference available: {card.inference_available}")
    except Exception as exc:
        print(f"  model card      FAIL {exc}")
        return 1

    if not card.confound_disclosure.statement:
        failures.append("model card carries no confound statement")
    if not card.confound_disclosure.severity_by_condition:
        failures.append("model card declares no per-condition confound severity")

    # --- cohort index ----------------------------------------------------
    try:
        subjects = cohort.load_cohort_index()
        print(f"  cohort index    OK   {len(subjects)} subjects")
    except Exception as exc:
        print(f"  cohort index    FAIL {exc}")
        return 1

    by_class: dict[str, int] = {}
    for subject in subjects:
        by_class[subject.true_class] = by_class.get(subject.true_class, 0) + 1
    print(f"                       per class: {by_class}")

    # --- every report parses against the response schema ------------------
    unreadable, softmax_like, fixtures = [], 0, 0
    for subject in subjects:
        try:
            report = cohort.get_report(subject.subject_id)
        except Exception as exc:
            unreadable.append(f"{subject.subject_id}: {exc}")
            continue
        total = sum(report.risk_scores.values())
        if abs(total - 1.0) < 1e-3:
            softmax_like += 1
        raw = json.loads(
            (Path(settings.eeg_model_dir) / "cohort" / f"{subject.subject_id}.json")
            .read_text(encoding="utf-8")
        )
        if raw.get("fixture"):
            fixtures += 1

    if unreadable:
        failures.append(f"{len(unreadable)} report(s) do not match the schema")
        for line in unreadable[:5]:
            print(f"    {line}")
    print(f"  reports parse   {'OK  ' if not unreadable else 'FAIL'} "
          f"{len(subjects) - len(unreadable)}/{len(subjects)}")

    # The single most likely silent regression.
    if softmax_like == len(subjects) and subjects:
        failures.append(
            "every subject's risk scores sum to 1.0 — these must be independent "
            "sigmoids, not a softmax distribution"
        )
    print(f"  independence    {'OK  ' if softmax_like < len(subjects) else 'FAIL'} "
          f"{len(subjects) - softmax_like}/{len(subjects)} subjects have non-unit score sums")

    # --- embeddings and projection ---------------------------------------
    try:
        vector = cohort.get_embedding(subjects[0].subject_id)
        norm_ok = abs(vector.l2_norm - 1.0) < 1e-2
        if not norm_ok:
            failures.append(f"z_eeg for {subjects[0].subject_id} is not unit-norm")
        print(f"  embeddings      {'OK  ' if norm_ok else 'FAIL'} "
              f"{vector.dim}-D, L2={vector.l2_norm:.4f}")
    except Exception as exc:
        warnings.append(f"embeddings unavailable: {exc}")
        print(f"  embeddings      WARN {exc}")

    try:
        projection = cohort.get_projection()
        covered = {p.subject_id for p in projection.points}
        missing = len({s.subject_id for s in subjects} - covered)
        if missing:
            warnings.append(f"{missing} subject(s) absent from the projection")
        print(f"  projection      OK   {len(projection.points)} points ({projection.method})")
    except Exception as exc:
        warnings.append(f"projection unavailable: {exc}")
        print(f"  projection      WARN {exc}")

    # --- band reference ---------------------------------------------------
    # Checked for shape and for whether it still discriminates at all: a reference
    # that reports a signature for every condition has almost certainly lost the
    # honesty gate, since one condition here separates only by an age artefact.
    try:
        reference = cohort.get_band_reference()
        verdicts = {
            name: ("signature" if profile.has_signature else "no signature")
            for name, profile in reference.conditions.items()
        }
        missing_bands = [
            name for name, profile in reference.conditions.items()
            if set(profile.auc_vs_hc) != set(reference.bands)
        ]
        if missing_bands:
            failures.append(f"band reference incomplete for: {', '.join(missing_bands)}")
        if all(p.has_signature for p in reference.conditions.values()):
            warnings.append(
                "every condition reports a band-power signature — verify the gate in "
                "services/eeg_band_statistics is still being applied"
            )
        print(f"  band reference  OK   HC n={reference.healthy.n}, "
              + ", ".join(f"{k} {v}" for k, v in verdicts.items()))
    except Exception as exc:
        warnings.append(f"band reference unavailable: {exc}")
        print(f"  band reference  WARN {exc}")

    # --- cohort/model provenance -----------------------------------------
    if fixtures:
        warnings.append(
            f"{fixtures}/{len(subjects)} reports are development FIXTURES with synthetic "
            "scores. Rebuild with --from-workspace before reporting anything."
        )

    print()
    for warning in warnings:
        print(f"  WARN  {warning}")
    for failure in failures:
        print(f"  FAIL  {failure}")

    print("\n" + "=" * 72)
    if failures:
        print(f"VERIFICATION FAILED — {len(failures)} problem(s)")
        print("=" * 72)
        return 1
    print("VERIFICATION PASSED" + (f" with {len(warnings)} warning(s)" if warnings else ""))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
