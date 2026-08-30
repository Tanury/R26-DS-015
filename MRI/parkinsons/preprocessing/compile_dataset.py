"""

Reads:
  - data/compiled/idaSearch_7_07_2026.csv       (LONI MRI collection export)
  - data/compiled/controlpdmixed.csv             (LONI DaTscan reconstructed collection)
  - data/nifti/mri/                              (MRI NIfTI files — from Colab preprocessing)
  - data/preprocessed/dat_slices_clean/          (DaTscan .npy slices — already done locally)
  - logs/datscan_excluded_subjects.txt           (quality filter — 77 bad subjects)

Writes:
  - data/compiled/labels.csv  with columns:
      subject_id | label | source | modality | file_path

Notes:
  - MRI entries point to NIfTI files (preprocessed by Colab)
  - DaTscan entries point to .npy slice files (preprocessed locally)
  - DaTscan quality-flagged subjects are excluded automatically
  
"""

import csv
import logging
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ = parkinsons/preprocessing/01_compile_dataset.py
# parents[0] = preprocessing/
# parents[1] = parkinsons/  ← ROOT
ROOT = Path(__file__).resolve().parents[1]

# Input CSVs
MRI_CSV     = ROOT / "data" / "compiled" / "idaSearch_7_07_2026.csv"
DAT_CSV     = ROOT / "data" / "compiled" / "controlpdmixed.csv"

# Input data dirs
MRI_NIFTI   = ROOT / "data" / "nifti" / "mri"                    # Colab output
DAT_SLICES  = ROOT / "data" / "preprocessed" / "dat_slices_clean" # local output

# Quality filter
EXCLUDE_TXT = ROOT / "logs" / "datscan_excluded_subjects.txt"

# Output
OUT_CSV     = ROOT / "data" / "compiled" / "labels.csv"

# ── MRI descriptions to reject ─────────────────────────────────────────────────
REJECT_MRI_DESCRIPTIONS = {
    "AX T1", "AX T1 SE C+", "TRA", "Transverse", "tra_T1_MPRAGE",
    "ax t1 reformat", "AX T1 reformat", "Coronal", "COR",
    "T1W_3D_FFE COR", "3D T1-weighted_MPR_cor", "3D T1-weighted_MPR_tra",
    "T1W_3D_FFE AX", "Cal Head 24", "SAG 3D MPRAGE RPT COIL ARTIFACT",
    "STRUC BRAVO SAG3D ARC", "sag 3D FSPGR BRAVO straight",
    "SAG 3D FSPGR BRAVO STRAIGHT", "Straight sagittal 3D T1w ADNI",
    "AX 3D FSPGR straight brain lab", "MPR - SmartBrain",
}

# ── Label mapping ──────────────────────────────────────────────────────────────
GROUP_MAP = {
    "PD":              "PD",
    "Control":         "HC",
    "HC":              "HC",
    "GenCohort PD":    "PD",
    "GenReg PD":       "PD",
    "Prodromal":       "PD",
    "SWEDD":           None,    # exclude — ambiguous diagnosis
    "Phantom":         None,
    "Volunteer":       None,
    "GenReg Unaff":    "HC",
    "GenCohort Unaff": "HC",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_excluded_subjects() -> set:
    """Load the 77 DaTscan subjects flagged by quality check."""
    excluded = set()
    if not EXCLUDE_TXT.exists():
        log.warning(f"Exclusion list not found: {EXCLUDE_TXT}")
        log.warning("Run preprocessing/dat/03_visual_check.py first")
        return excluded

    with open(EXCLUDE_TXT) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            subject_id = line.split(',')[0].strip()
            excluded.add(subject_id)

    log.info(f"Loaded {len(excluded)} excluded DaTscan subjects from quality filter")
    return excluded


def load_mri_csv() -> dict:
    """
    Returns {subject_id: {"label": str}}
    Source: idaSearch_7_07_2026.csv (LONI MRI expanded collection)
    Only Subject ID, Description columns available in this export.
    Note: no Group column — MRI CSV from idaSearch only has Subject ID + Description.
    We derive labels from controlpdmixed.csv cross-reference instead.
    """
    if not MRI_CSV.exists():
        log.warning(f"MRI CSV not found: {MRI_CSV}")
        return {}

    # Load DaTscan CSV first to get group labels (same subjects appear in both)
    dat_labels = {}
    if DAT_CSV.exists():
        with open(DAT_CSV, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj  = row.get("Subject", "").strip()
                group = row.get("Group", "").strip()
                label = GROUP_MAP.get(group)
                if subj and label:
                    dat_labels[subj] = label

    records = {}
    with open(MRI_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row.get("Subject ID", row.get("Subject", "")).strip()
            desc = row.get("Description", "").strip()

            if not subj or desc in REJECT_MRI_DESCRIPTIONS:
                continue

            # Use label from DaTscan CSV if available, otherwise skip
            # (MRI CSV doesn't have Group column in idaSearch export)
            label = dat_labels.get(subj)
            if label is None:
                continue

            if subj not in records:
                records[subj] = {"label": label}

    log.info(f"MRI CSV: {len(records)} usable subjects loaded")
    return records


def load_dat_csv(excluded: set) -> dict:
    """
    Returns {subject_id: {"label": str}}
    Source: controlpdmixed.csv (1,946 reconstructed DaTscan subjects)
    Applies quality exclusion list.
    """
    if not DAT_CSV.exists():
        log.warning(f"DaTscan CSV not found: {DAT_CSV}")
        return {}

    records   = {}
    n_excluded = 0

    with open(DAT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj  = row.get("Subject", row.get("Subject ID", "")).strip()
            group = row.get("Group", "").strip()

            if not subj:
                continue

            # Apply quality filter
            if subj in excluded:
                n_excluded += 1
                continue

            label = GROUP_MAP.get(group)
            if label is None:
                continue

            if subj not in records:
                records[subj] = {"label": label}

    log.info(f"DaTscan CSV: {len(records)} usable subjects loaded ({n_excluded} excluded by quality filter)")
    return records


def find_mri_nifti(subject_id: str) -> Path | None:
    """Find preprocessed MRI NIfTI — Colab names: <subject_id>_reg_brain_n4.nii.gz"""
    candidates = [
        MRI_NIFTI / f"{subject_id}_reg_brain_n4.nii.gz",
        MRI_NIFTI / f"{subject_id}.nii.gz",
        MRI_NIFTI / f"sub-{subject_id}.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(MRI_NIFTI.glob(f"*{subject_id}*.nii.gz"))
    return matches[0] if matches else None


def find_dat_slice(subject_id: str) -> Path | None:
    """Find preprocessed DaTscan slice — named: <subject_id>_slice.npy"""
    p = DAT_SLICES / f"{subject_id}_slice.npy"
    return p if p.exists() else None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("01_compile_dataset.py — building labels.csv")
    log.info("=" * 60)
    log.info(f"ROOT: {ROOT}")

    excluded    = load_excluded_subjects()
    mri_records = load_mri_csv()
    dat_records = load_dat_csv(excluded)

    rows  = []
    stats = defaultdict(int)

    # ── MRI subjects ──────────────────────────────────────────────────────────
    for subj, meta in mri_records.items():
        file_path = find_mri_nifti(subj)
        if file_path is None:
            stats["mri_file_missing"] += 1
            continue

        rows.append({
            "subject_id": subj,
            "label":      meta["label"],
            "source":     "PPMI",
            "modality":   "MRI",
            "file_path":  str(file_path.relative_to(ROOT)),
        })
        stats[f"mri_{meta['label']}"] += 1

    # ── DaTscan subjects ──────────────────────────────────────────────────────
    for subj, meta in dat_records.items():
        file_path = find_dat_slice(subj)
        if file_path is None:
            stats["dat_file_missing"] += 1
            continue

        rows.append({
            "subject_id": subj,
            "label":      meta["label"],
            "source":     "PPMI",
            "modality":   "DaTscan",
            "file_path":  str(file_path.relative_to(ROOT)),
        })
        stats[f"dat_{meta['label']}"] += 1

    # ── Write CSV ─────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "label", "source", "modality", "file_path"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Report ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("── Results ──────────────────────────────────────────")
    log.info(f"  MRI   PD           : {stats['mri_PD']}")
    log.info(f"  MRI   HC           : {stats['mri_HC']}")
    log.info(f"  MRI   file missing : {stats['mri_file_missing']}  (run Colab MRI preprocessing)")
    log.info(f"  DaTscan PD         : {stats['dat_PD']}")
    log.info(f"  DaTscan HC         : {stats['dat_HC']}")
    log.info(f"  DaTscan missing    : {stats['dat_file_missing']}")
    log.info(f"  Total rows written : {len(rows)}")
    log.info(f"  Output             : {OUT_CSV}")
    log.info("")

    mri_subjs = {r["subject_id"] for r in rows if r["modality"] == "MRI"}
    dat_subjs = {r["subject_id"] for r in rows if r["modality"] == "DaTscan"}
    paired    = mri_subjs & dat_subjs
    log.info(f"  Paired (MRI + DaTscan): {len(paired)}")


if __name__ == "__main__":
    main()