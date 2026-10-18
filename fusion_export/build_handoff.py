"""
build_handoff.py
================
Writes outputs/fusion_export/metadata.json and README.md from the per-branch
metrics.json files. Re-run after any branch export changes.

    python fusion_export/build_handoff.py
"""

import json
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs/fusion_export"

# Which probability column the fusion engine should use, and why.
USE = {
    "pd_datscan": ("predictions.csv", "P_PD_raw", "test_raw",
                   "Raw probabilities are already well calibrated (Brier 0.071); "
                   "calibration did not improve them."),
    "ms_mri": ("predictions_oof.csv", "P_MS_calibrated_oof", "loocv_calibrated",
               "Calibration lowered Brier and raised specificity; the raw model over-calls MS."),
    "ms_oct": ("predictions_oof.csv", "P_MS_raw_oof", "loocv_raw",
               "Near chance. Calibrated values collapse towards the base rate, so use raw "
               "with a very low weight, or drop this branch."),
}
DISEASE = {"pd_datscan": "PD", "ms_mri": "MS", "ms_oct": "MS"}


def load(b):
    return json.loads((OUT / b / "metrics.json").read_text())


def main():
    m = {b: load(b) for b in USE}

    # Suggested starting weights within each disease: proportional to (AUC - 0.5).
    weights = {}
    for dis in set(DISEASE.values()):
        bs = [b for b in USE if DISEASE[b] == dis]
        skill = {b: max(m[b][USE[b][2]]["roc_auc"] - 0.5, 0) for b in bs}
        tot = sum(skill.values()) or 1
        for b in bs:
            weights[b] = round(skill[b] / tot, 3)

    meta = {
        "component": "R26-DS-015 vision encoder (Dissanayaka T.)",
        "for": "decision-level fusion with missing-modality handling",
        "exported": str(date.today()),
        "embedding": {"dim": 256, "normalisation": "L2 (unit norm)", "file": "z_img.npz"},
        "branches": {},
        "not_included": {
            "ad_mri": "On hold. Training inputs mixed brain images and brain masks and the "
                      "test split was by file, not subject; results not reliable yet.",
            "pd_mri": "Below chance (AUC ~0.47). Not for fusion.",
        },
    }
    for b, (f, col, key, why) in USE.items():
        s = m[b][key]
        meta["branches"][b] = {
            "disease": DISEASE[b],
            "modality": m[b]["modality"],
            "dataset": m[b]["dataset"],
            "n": m[b]["n"],
            "classes": ["HC", DISEASE[b]],
            "predictions_file": f"{b}/{f}",
            "use_probability_column": col,
            "why": why,
            "evaluation": m[b]["evaluation"],
            "metrics_for_that_column": {k: round(v, 4) for k, v in s.items()},
            "majority_class_rate": round(m[b]["majority_class_rate"], 4),
            "suggested_weight_within_disease": weights[b],
            "platt_for_new_patients": m[b]["platt"],
        }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))

    def row(b):
        s = m[b][USE[b][2]]
        return (f"| `{b}` | {DISEASE[b]} | {m[b]['n']} | {s['accuracy']*100:.1f}% | {s['roc_auc']:.3f} "
                f"| {s[[k for k in s if k.startswith('sensitivity')][0]]*100:.0f}% "
                f"| {s[[k for k in s if k.startswith('specificity')][0]]*100:.0f}% "
                f"| `{USE[b][1]}` | {weights[b]} |")

    mri_imp = m["ms_mri"]["global_feature_importance_mean_abs_attr"]
    top_mri = max(mri_imp, key=mri_imp.get).replace("_frac", "")

    readme = f"""# Vision encoder outputs for the fusion engine

From: Tanuri (vision encoder, R26-DS-015)
Exported: {date.today()}

Three imaging branches are ready. Each one only predicts its own disease against
healthy controls. Use them for decision-level fusion: combine the probabilities,
weighted by how reliable each branch is.

## Branches at a glance

| Branch | Disease | n | Accuracy | ROC-AUC | Sensitivity | Specificity | Use column | Suggested weight* |
|---|---|---|---|---|---|---|---|---|
{row("pd_datscan")}
{row("ms_mri")}
{row("ms_oct")}

*Weight within that disease, proportional to (AUC − 0.5). PD has one imaging branch,
so its weight is 1. This is a starting point; tune it on your side.

Reliability, strongest first: **PD-DaTscan → MS-MRI → MS-OCT (near chance)**.

## Files in each branch folder

| File | What it is |
|---|---|
| `predictions.csv` / `predictions_oof.csv` | One row per subject: `subject_id`, `label` (1 = disease, 0 = HC), raw and calibrated probability, `pred` |
| `z_img.npz` | 256-d L2-normalised embeddings, with `subject_ids` and `labels` in the same order |
| `metrics.json` | Full metrics, evaluation method, and Platt calibration parameters |
| `features.csv` (MS only) | The exact input features per subject (unscaled) |
| `attributions.csv` (MS only) | How much each feature pushed P(MS) up or down for each subject |

`metadata.json` in this folder collects the key facts for all branches.

## How the probabilities were produced

- **MS-MRI and MS-OCT:** leave-one-out cross-validation. Every probability comes from a
  model that never saw that subject, so they are safe to learn fusion weights from.
- **PD-DaTscan:** a single held-out test set (282 PPMI subjects, one scan each).
- **Calibration:** Platt scaling, fitted leave-one-out so the calibrated values are also
  out-of-sample. For a new patient, apply `P_cal = sigmoid(a * logit(P_raw) + b)` with
  `a`, `b` from `metrics.json` → `platt`.
- **Embeddings:** MS embeddings come from the production model fitted on all subjects,
  so they are in-sample for these subjects. PD embeddings are from the test set.

## Things to handle on the fusion side

1. **Missing modalities.** When a branch is absent for a patient, renormalise the weights
   over the branches that are present. Do not treat a missing branch as probability 0.
2. **No shared patients.** PPMI, MS3SEG/LEMON and the JHU OCT set are different people,
   and they do not overlap with the EEG or biomarker cohorts. There is no patient with
   all modalities, so weights cannot be learned on matched multimodal patients.
3. **Each branch only knows its own disease.** The MS branches never saw a PD patient, so
   their P(HC) says nothing about PD. Produce one risk score per disease rather than one
   combined diagnosis.
4. **MS-MRI over-calls MS:** specificity is about 70%, so roughly 3 in 10 healthy controls
   score as MS. Its strongest feature by far is {top_mri.lower()} volume.
5. **MS-OCT is near chance** (35 subjects). Give it a very low weight or leave it out;
   dropping it costs little.

## Not included

- **AD (MRI):** on hold. The training inputs mixed brain images with brain masks, and the
  test split was by file rather than subject, so the current results are not reliable.
- **PD (structural MRI):** below chance (AUC ~0.47). Not for fusion.

Questions: ask Tanuri.
"""
    (OUT / "README.md").write_text(readme)
    print(readme)


if __name__ == "__main__":
    main()
