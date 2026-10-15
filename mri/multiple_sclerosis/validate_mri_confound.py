"""
validate_mri_confound.py
==========================
R26-DS-015 -- MRI-MS branch (MindGlide-based)

IMPORTANT CONTEXT FIRST: in this dataset, cohort and label are the SAME
split -- every label=1 (MS) subject comes from MS3SEG and every label=0
(HC) subject comes from LEMON. There is no subject with label=1 scanned
under the LEMON protocol or label=0 under the MS3SEG protocol. That means
you CANNOT directly separate "is the model detecting MS pathology" from
"is the model detecting which scanner/study this came from" using
classification accuracy alone -- the two explanations predict identical
accuracy numbers. This is a real, unresolved limitation of the dataset,
not something this script fixes. What it CAN do is tell you how big the
risk is, via one concrete test:

  Pick regions with little-to-no established MS association (the
  "neutral" set below -- cerebellum, pons, brainstem, occipital lobe GM,
  temporal lobe) and test whether THEY also differ significantly between
  the HC (LEMON) and MS (MS3SEG) groups, using the same normalize-by-
  total-brain-volume treatment as your real feature set.

  - If the neutral regions do NOT differ significantly -> the cohort/
    scanner effect is probably small; your classifier's signal on the
    literature-subset regions (Lesion, Lateral_ventricle, DGM,
    Corpus_callosum, White_matter) is more likely genuine disease signal.
  - If the neutral regions DO differ significantly -> there's a
    systematic, non-biological difference between the two cohorts large
    enough that MindGlide's volume estimates pick it up regardless of
    disease status. Report this explicitly as a limitation: some of the
    reported accuracy may reflect acquisition/protocol differences
    between MS3SEG and LEMON rather than MS pathology.

Uses the Mann-Whitney U test (non-parametric, appropriate for n~35-150
and no assumption of normality) with Bonferroni-corrected significance
to avoid false positives from testing many regions at once.

Usage:
    python3 validate_mri_confound.py mindglide_all_features.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RAW_REGIONS_FOR_TOTAL = [
    "CSF", "Ventricles_3_4_5", "DGM", "Pons", "Brainstem", "Cerebellum",
    "Temporal_lobe", "Temporal_horn_lateral_ventricle", "Lateral_ventricle",
    "Optic_chiasm", "Cerebellar_vermis", "Corpus_callosum", "White_matter",
    "Frontal_lobe_GM", "Limbic_cortex_GM", "Parietal_lobe_GM",
    "Occipital_lobe_GM", "Lesion", "Ventral_diencephalon",
]
LITERATURE_SUBSET = ["Lesion", "Lateral_ventricle", "DGM", "Corpus_callosum", "White_matter"]

# Regions with little established MS-specific association in the
# literature -- used here only as a confound probe, not as clinical claims.
NEUTRAL_SUBSET = ["Pons", "Brainstem", "Cerebellum", "Occipital_lobe_GM", "Temporal_lobe"]


def prepare_fractions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_brain_volume"] = df[RAW_REGIONS_FOR_TOTAL].sum(axis=1)
    for r in RAW_REGIONS_FOR_TOTAL:
        df[r + "_frac"] = df[r] / df["total_brain_volume"]
    return df


def compare_groups(df: pd.DataFrame, regions: list, n_tests_for_correction: int) -> pd.DataFrame:
    rows = []
    hc = df[df["label"] == 0]
    ms = df[df["label"] == 1]
    for r in regions:
        col = r + "_frac"
        u_stat, p_raw = mannwhitneyu(hc[col], ms[col], alternative="two-sided")
        p_corrected = min(p_raw * n_tests_for_correction, 1.0)  # Bonferroni
        rows.append({
            "region": r,
            "hc_mean_frac": round(hc[col].mean(), 5),
            "ms_mean_frac": round(ms[col].mean(), 5),
            "p_raw": round(p_raw, 5),
            "p_bonferroni": round(p_corrected, 5),
            "significant_at_0.05": p_corrected < 0.05,
        })
    return pd.DataFrame(rows)


def main(csv_path: str) -> None:
    raw_df = pd.read_csv(csv_path, dtype={"id": str})
    df = prepare_fractions(raw_df)

    n_hc = (df["label"] == 0).sum()
    n_ms = (df["label"] == 1).sum()
    print(f"Loaded {len(df)} subjects: {n_hc} HC (LEMON) + {n_ms} MS (MS3SEG)\n")
    print("Reminder: cohort == label exactly in this dataset. The test below measures")
    print("how much of ANY group difference could plausibly be scanner/protocol driven,")
    print("not a direct cohort-vs-label comparison (that comparison doesn't exist here).\n")

    all_regions = sorted(set(LITERATURE_SUBSET + NEUTRAL_SUBSET))
    n_tests = len(all_regions)  # Bonferroni across everything tested in this run

    print("=" * 78)
    print("LITERATURE-SUBSET REGIONS (expected to differ -- this is your real signal)")
    print("=" * 78)
    lit_results = compare_groups(df, LITERATURE_SUBSET, n_tests)
    print(lit_results.to_string(index=False))

    print("\n" + "=" * 78)
    print("NEUTRAL REGIONS (little MS-specific association -- confound probe)")
    print("=" * 78)
    neutral_results = compare_groups(df, NEUTRAL_SUBSET, n_tests)
    print(neutral_results.to_string(index=False))

    n_neutral_sig = neutral_results["significant_at_0.05"].sum()
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    if n_neutral_sig == 0:
        print(f"  None of the {len(NEUTRAL_SUBSET)} neutral regions differ significantly "
              f"between groups after Bonferroni correction.")
        print("  This is reassuring: the two cohorts look broadly comparable on regions ")
        print("  with no expected disease effect, which weakens (does not eliminate) the ")
        print("  scanner/protocol-confound explanation for your classifier's accuracy.")
    else:
        print(f"  {n_neutral_sig} of {len(NEUTRAL_SUBSET)} neutral regions DO differ ")
        print("  significantly between HC (LEMON) and MS (MS3SEG) after correction.")
        print("  Since these regions have no established MS association, this points to a")
        print("  systematic acquisition/protocol difference between the two studies, not")
        print("  disease biology. State this explicitly as a limitation in the paper: some")
        print("  portion of the reported MRI-branch accuracy may reflect which scanner/")
        print("  cohort a subject came from rather than MS pathology. This cannot be fully")
        print("  resolved without MS patients scanned under the LEMON protocol (or vice")
        print("  versa) -- flag it as future work rather than something to tune away.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])