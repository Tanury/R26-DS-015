"""
compile.py - step 1 of preprocessing pipeline for Alzheimer's MRI data

what it does:
reads ADNI metadata CSV files
matchess each NIfTI file to its class label (CN, MCI, AD)
copies each file into a new sorted subdirectory (CN, MCI, AD) in data/sorted
generates summary of no.of files sorted per class

"""

import os
import argparse
import shutil
import pandas as pd
from pathlib import Path

DEFAULT_CSV_PATH = "MRI/alzheimers/data/raw/metadata.csv"
DEFAULT_NIFTI_DIR  = "mri/alzheimers/data/raw"
DEFAULT_OUTPUT_DIR = "mri/alzheimers/data/sorted"

VALID_LABELS = ["AD","MCI","CN"]

COL_SUBJECT_ID = "Subject ID"
COL_IMAGE_ID   = "Image Data ID"
COL_LABEL      = "Diagnosis"    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort ADNI NIfTI files into class based subdirectories"
     )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--nifti_dir", type=str, default=DEFAULT_NIFTI_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()

def load_metadata(csv_path:str) -> pd.DataFrame:
    print(f"\n[1/5] Loading metadata from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'.\n"
            f"Place ADNI metadata CSV at :\n"
            f"  R26-DS-015/mri/alzheimers/data/raw/metadata.csv"
        )
    df = pd.read_csv(csv_path)
    print(f"Metadata loaded. Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    for col in [COL_SUBJECT_ID, COL_IMAGE_ID, COL_LABEL]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV.\n"
                f"Update the column name constants at the top of this script.\n"
                f"Available columns: {list(df.columns)}"
            )
 
    original_count = len(df)
    df = df[df[COL_LABEL].isin(VALID_LABELS)].copy()
    assert isinstance(df, pd.DataFrame)
    dropped = original_count - len(df)
    if dropped > 0:
        print(f"      Dropped {dropped} rows with unrecognised labels.")
    print(f"      Valid rows: {len(df)}")
    return df

def discover_nifti_files(nifti_dir: str) -> dict:
    print(f"\n[2/5] Scanning for NIfTI files in: {nifti_dir}")
 
    if not os.path.exists(nifti_dir):
        raise FileNotFoundError(
            f"Directory not found: '{nifti_dir}'.\n"
            f"Place raw .nii/.nii.gz files in:\n"
            f"  R26-DS-015/mri/alzheimers/data/raw/"
        )
 
    nifti_map  = {}
    extensions = (".nii", ".nii.gz")
 
    for root, _, files in os.walk(nifti_dir):
        for filename in files:
            if any(filename.endswith(ext) for ext in extensions):
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
    print(f"\n[3/5] Creating output directories under: {output_dir}")
    label_dirs = {}
    for label in VALID_LABELS:
        label_path = Path(output_dir) / label
        label_path.mkdir(parents=True, exist_ok=True)
        label_dirs[label] = label_path
        print(f"      Created: {label_path}")
    return label_dirs


def sort_files(df: pd.DataFrame, nifti_map: dict, label_dirs: dict) -> dict:
    print(f"\n[4/5] Sorting files...")
 
    summary   = {label: 0 for label in VALID_LABELS}
    not_found = []
    skipped   = []
 
    for _, row in df.iterrows():
        image_id = str(row[COL_IMAGE_ID]).strip()
        label    = str(row[COL_LABEL]).strip()
 
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
        print(f"\n  ⚠️  Could not match {len(not_found)} Image IDs to files.")
        print(f"      First 10: {not_found[:10]}")
        print(f"      Normal if you haven't downloaded all ADNI images yet.")
 
    if skipped:
        print(f"\n  ℹ️   Skipped {len(skipped)} already-existing files (re-run safe).")
 
    return summary

def print_summary(summary: dict, output_dir: str):
    print(f"\n[5/5] Sorting complete. Summary:")
    print(f"      {'─' * 30}")
    total = 0
    for label, count in summary.items():
        print(f"      {label:<6} : {count} files")
        total += count
    print(f"      {'─' * 30}")
    print(f"      Total  : {total} files")
    print(f"\n      Output: {os.path.abspath(output_dir)}")
    print(f"\n      ✅ Next step: run register.py\n")

def main():
    args = parse_args()

    print("=" * 55)
    print(" Alzheimer's MRI Preprocessing - Step 1: Compile & Sort")
    print("=" * 55)

    df = load_metadata(args.csv)
    nifti_map = discover_nifti_files(args.nifti_dir)
    label_dirs = create_output_dirs(args.output_dir)
    summary = sort_files(df, nifti_map, label_dirs)
    print_summary(summary, args.output_dir)

if __name__ == "__main__":
    main()
    
