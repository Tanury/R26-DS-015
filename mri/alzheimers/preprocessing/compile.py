"""
compile.py - Step 1 of Alzheimer's MRI Preprocessing Pipeline

What it does:
    Reads ADNI_NIfTI.csv 
    Reads ADNIMERGE.csv   
    Joins them on Subject ID to get a label for each image you downloaded
    Matches each NIfTI file to its label (CN, MCI, AD)
    Copies files into data/sorted/{AD,MCI,CN}/
    Prints a summary

Why two CSVs:
    ADNI_NIfTI.csv knows WHICH images you downloaded and their Image IDs
    ADNIMERGE.csv  knows the DIAGNOSIS for each subject
    Neither alone is sufficient — joining them gives label per image

Expected file locations:
    mri/alzheimers/data/raw/collection.csv   ← rename ADNI_NIfTI.csv to this
    mri/alzheimers/data/raw/metadata.csv     ← ADNIMERGE.csv
    mri/alzheimers/data/raw/*.nii.gz         ← converted NIfTI files

ADNIMERGE label normalisation:
    CN        → CN
    MCI       → MCI
    LMCI      → MCI
    EMCI      → MCI
    Dementia  → AD
    AD        → AD
"""

import os
import argparse
import shutil
import pandas as pd
from pathlib import Path

DEFAULT_COLLECTION_CSV = "mri/alzheimers/data/raw/collection.csv"
DEFAULT_ADNIMERGE_CSV  = "mri/alzheimers/data/raw/metadata.csv"
DEFAULT_NIFTI_DIR      = "mri/alzheimers/data/raw"
DEFAULT_OUTPUT_DIR     = "mri/alzheimers/data/sorted"

VALID_LABELS = ["AD", "MCI", "CN"]

# ADNI_NIfTI.csv columns
COL_COLLECTION_IMAGE_ID  = "Image Data ID"   # e.g. I238622
COL_COLLECTION_SUBJECT   = "Subject"         # e.g. 002_S_0295

# ADNIMERGE.csv columns
COL_MERGE_SUBJECT        = "PTID"            # e.g. 002_S_0295
COL_MERGE_LABEL          = "DX_bl"           # baseline diagnosis

LABEL_MAP = {
    "CN":       "CN",
    "MCI":      "MCI",
    "LMCI":     "MCI",
    "EMCI":     "MCI",
    "Dementia": "AD",
    "AD":       "AD",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort ADNI NIfTI files into class-based subdirectories."
    )
    parser.add_argument("--collection_csv", type=str, default=DEFAULT_COLLECTION_CSV,
                        help="Path to ADNI_NIfTI.csv (IDA collection export).")
    parser.add_argument("--adnimerge_csv",  type=str, default=DEFAULT_ADNIMERGE_CSV,
                        help="Path to ADNIMERGE.csv.")
    parser.add_argument("--nifti_dir",      type=str, default=DEFAULT_NIFTI_DIR)
    parser.add_argument("--output_dir",     type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_collection(csv_path: str) -> pd.DataFrame:
    print(f"\n[1/6] Loading collection CSV: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Collection CSV not found at '{csv_path}'.\n"
            f"Rename your ADNI_NIfTI.csv to collection.csv and place it at:\n"
            f"  R26-DS-015/mri/alzheimers/data/raw/collection.csv"
        )
    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df)} rows.")

    for col in [COL_COLLECTION_IMAGE_ID, COL_COLLECTION_SUBJECT]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in collection CSV.\n"
                f"Available: {list(df.columns)}"
            )

    # Normalise Image ID — strip quotes/whitespace, ensure "I" prefix
    df[COL_COLLECTION_IMAGE_ID] = df[COL_COLLECTION_IMAGE_ID].astype(str).str.strip().str.strip('"')
    df[COL_COLLECTION_SUBJECT]  = df[COL_COLLECTION_SUBJECT].astype(str).str.strip().str.strip('"')
    return df[[COL_COLLECTION_IMAGE_ID, COL_COLLECTION_SUBJECT]]


def load_adnimerge(csv_path: str) -> pd.DataFrame:
    print(f"\n[2/6] Loading ADNIMERGE: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ADNIMERGE not found at '{csv_path}'.\n"
            f"Download ADNIMERGE.csv from IDA → Download → Study Data → ADNIMERGE\n"
            f"Place it at: R26-DS-015/mri/alzheimers/data/raw/metadata.csv"
        )
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"      Loaded {len(df)} rows.")

    for col in [COL_MERGE_SUBJECT, COL_MERGE_LABEL]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in ADNIMERGE.\n"
                f"Available: {list(df.columns)}"
            )

    df[COL_MERGE_SUBJECT] = df[COL_MERGE_SUBJECT].astype(str).str.strip()

    # Keep only one row per subject (baseline) — ADNIMERGE has multiple visits
    df = df[[COL_MERGE_SUBJECT, COL_MERGE_LABEL]].drop_duplicates(
        subset=COL_MERGE_SUBJECT, keep="first"
    )
    print(f"      Unique subjects: {len(df)}")
    return df


def build_image_label_map(collection: pd.DataFrame, adnimerge: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3/6] Joining collection with ADNIMERGE on Subject ID...")

    merged = collection.merge(
        adnimerge,
        left_on=COL_COLLECTION_SUBJECT,
        right_on=COL_MERGE_SUBJECT,
        how="left"
    )

    # Normalise labels
    merged["_label"] = merged[COL_MERGE_LABEL].map(LABEL_MAP)

    before = len(merged)
    merged = merged[merged["_label"].isin(VALID_LABELS)].copy()
    dropped = before - len(merged)

    if dropped > 0:
        print(f"      Dropped {dropped} rows with missing/unrecognised diagnosis.")
    print(f"      Matched {len(merged)} images with valid labels.")

    # Show class distribution
    dist = merged["_label"].value_counts()
    for label in VALID_LABELS:
        print(f"        {label:<6}: {dist.get(label, 0)}")

    return merged[[COL_COLLECTION_IMAGE_ID, "_label"]]


def discover_nifti_files(nifti_dir: str) -> dict:
    print(f"\n[4/6] Scanning NIfTI files in: {nifti_dir}")
    if not os.path.exists(nifti_dir):
        raise FileNotFoundError(f"Directory not found: '{nifti_dir}'")

    nifti_map = {}
    for root, _, files in os.walk(nifti_dir):
        for filename in files:
            if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                filepath = Path(root) / filename
                stem     = filename.replace(".nii.gz", "").replace(".nii", "")
                parts    = stem.split("_")

                image_id = None
                for part in reversed(parts):
                    if part.startswith("I") and part[1:].isdigit():
                        image_id = part
                        break

                key = image_id if image_id else stem
                nifti_map[key] = filepath

    print(f"      Found {len(nifti_map)} NIfTI files.")
    return nifti_map


def create_output_dirs(output_dir: str) -> dict:
    print(f"\n[5/6] Creating output directories under: {output_dir}")
    label_dirs = {}
    for label in VALID_LABELS:
        label_path = Path(output_dir) / label
        label_path.mkdir(parents=True, exist_ok=True)
        label_dirs[label] = label_path
        print(f"      Created: {label_path}")
    return label_dirs


def sort_files(image_label_map: pd.DataFrame, nifti_map: dict, label_dirs: dict) -> dict:
    print(f"\n[6/6] Sorting files...")

    summary   = {label: 0 for label in VALID_LABELS}
    not_found = []
    skipped   = []

    for _, row in image_label_map.iterrows():
        image_id = str(row[COL_COLLECTION_IMAGE_ID]).strip()
        label    = str(row["_label"]).strip()

        if label not in VALID_LABELS:
            continue

        if image_id not in nifti_map:
            not_found.append(image_id)
            continue

        src_path  = nifti_map[image_id]
        dest_path = label_dirs[label] / src_path.name

        if dest_path.exists():
            skipped.append(str(dest_path))
            summary[label] += 1
            continue

        shutil.copy2(src_path, dest_path)
        summary[label] += 1

    if not_found:
        print(f"\n    Could not find NIfTI files for {len(not_found)} Image IDs.")
        print(f"      First 10: {not_found[:10]}")
        print(f"      These images may not have been downloaded yet.")

    if skipped:
        print(f"\n  Skipped {len(skipped)} already-existing files (re-run safe).")

    return summary


def print_summary(summary: dict, output_dir: str):
    print(f"\n{'-' * 35}")
    print(f"  Sorting Complete")
    print(f"  {'─' * 35}")
    total = 0
    for label, count in summary.items():
        print(f"  {label:<6} : {count} files")
        total += count
    print(f"  {'─' * 35}")
    print(f"  Total  : {total} files")
    print(f"\n  Output : {os.path.abspath(output_dir)}")
    print(f"\n  Step 2 : run register.py\n")


def main():
    args = parse_args()

    print("-" * 35)
    print("  Alzheimer's MRI Preprocessing - Step 1: Compile & Sort | Complete")
    print("-" * 35)

    collection     = load_collection(args.collection_csv)
    adnimerge      = load_adnimerge(args.adnimerge_csv)
    image_label_map = build_image_label_map(collection, adnimerge)
    nifti_map      = discover_nifti_files(args.nifti_dir)
    label_dirs     = create_output_dirs(args.output_dir)
    summary        = sort_files(image_label_map, nifti_map, label_dirs)
    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()