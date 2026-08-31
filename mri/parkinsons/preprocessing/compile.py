import os
import shutil
import argparse
import pandas as pd
from pathlib import Path


DEFAULT_OUTPUT_DIR = "mri/parkinsons/data/sorted"
CLASS_LABELS       = ["PD", "HC"]

# Common column names across datasets
SUBJECT_COL_CANDIDATES    = ["participant_id", "subject_id", "Subject", "ID", "sub"]
DIAGNOSIS_COL_CANDIDATES  = ["diagnosis", "group", "Group", "DX", "dx", "label",
                              "Diagnosis", "condition"]
PD_LABELS   = {"pd", "parkinson", "parkinson's", "patient", "1", "case"}
HC_LABELS   = {"hc", "control", "healthy", "normal", "nc", "0", "ctrl"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort Parkinson's NIfTI files into PD/HC class folders."
    )
    parser.add_argument("--nifti_dir",   type=str, required=True,
                        help="Directory containing downloaded NIfTI files (can be nested).")
    parser.add_argument("--metadata",    type=str, default=None,
                        help="Path to participants.tsv or CSV with subject ID and diagnosis columns.")
    parser.add_argument("--output_dir",  type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for sorted files.")
    parser.add_argument("--label",       type=str, default=None, choices=["PD", "HC"],
                        help="If all files in nifti_dir belong to one class, specify it here.")
    parser.add_argument("--copy",        action="store_true", default=True,
                        help="Copy files (default) instead of moving them.")
    parser.add_argument("--move",        action="store_true",
                        help="Move files instead of copying.")
    return parser.parse_args()


def find_nifti_files(root: str) -> list:
    """Recursively find all NIfTI files."""
    root_path = Path(root)
    files = list(root_path.rglob("*.nii.gz")) + list(root_path.rglob("*.nii"))
    # Exclude files with 'mask' or 'seg' in name — only want structural T1
    files = [f for f in files if not any(
        x in f.name.lower() for x in ["mask", "seg", "lesion", "label", "brain_mask"]
    )]
    return sorted(files)


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load participants metadata file (TSV or CSV)."""
    path = Path(metadata_path)
    if path.suffix == ".tsv":
        df = pd.read_csv(metadata_path, sep="\t")
    else:
        df = pd.read_csv(metadata_path)

    # Find subject and diagnosis columns
    sub_col = next((c for c in SUBJECT_COL_CANDIDATES if c in df.columns), None)
    dx_col  = next((c for c in DIAGNOSIS_COL_CANDIDATES if c in df.columns), None)

    if not sub_col:
        raise ValueError(
            f"Could not find subject ID column. Available: {list(df.columns)}\n"
            f"Expected one of: {SUBJECT_COL_CANDIDATES}"
        )
    if not dx_col:
        raise ValueError(
            f"Could not find diagnosis column. Available: {list(df.columns)}\n"
            f"Expected one of: {DIAGNOSIS_COL_CANDIDATES}"
        )

    df = df[[sub_col, dx_col]].copy()
    df.columns = ["subject_id", "diagnosis"]
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["diagnosis"]  = df["diagnosis"].astype(str).str.strip().str.lower()

    # Normalise diagnosis labels
    df["class"] = df["diagnosis"].apply(lambda d:
        "PD" if d in PD_LABELS else
        "HC" if d in HC_LABELS else None
    )
    unmapped = df[df["class"].isna()]["diagnosis"].unique()
    if len(unmapped) > 0:
        print(f"  ⚠️  Unmapped diagnosis values: {unmapped}")
        print(f"     Edit PD_LABELS/HC_LABELS in compile.py if needed")

    return df.dropna(subset=["class"])


def match_subject(nifti_path: Path, metadata: pd.DataFrame) -> str | None:
    """
    Try to match a NIfTI filename or parent directory to a subject ID in metadata.
    BIDS format: sub-{ID}/anat/sub-{ID}_T1w.nii.gz
    NITRC format: {SubjectID}_*.nii.gz or {SubjectID}/*.nii.gz
    """
    # Try parent directories first (BIDS sub-ID folders)
    for part in nifti_path.parts:
        part_clean = part.replace("sub-", "").strip()
        match = metadata[metadata["subject_id"].str.contains(
            part_clean, case=False, na=False, regex=False
        )]
        if len(match) == 1:
            return match.iloc[0]["class"]

    # Try filename
    stem = nifti_path.stem.replace(".nii", "")
    for _, row in metadata.iterrows():
        if row["subject_id"] in stem or stem in row["subject_id"]:
            return row["class"]

    return None


def main():
    args = parse_args()
    use_move = args.move

    print("\n" + "=" * 55)
    print("  Parkinson's MRI — Compile: Sort into PD/HC folders")
    print("=" * 55)

    # Create output directories
    for cls in CLASS_LABELS:
        Path(args.output_dir, cls).mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files
    print(f"\n  Scanning: {args.nifti_dir}")
    nifti_files = find_nifti_files(args.nifti_dir)
    print(f"  Found: {len(nifti_files)} NIfTI files")

    if len(nifti_files) == 0:
        print("  ❌  No NIfTI files found. Check --nifti_dir path.")
        return

    # Load metadata if provided
    metadata = None
    if args.metadata:
        print(f"\n  Loading metadata: {args.metadata}")
        metadata = load_metadata(args.metadata)
        print(f"  Subjects with labels: {len(metadata)}")
        print(f"  PD: {(metadata['class']=='PD').sum()} | HC: {(metadata['class']=='HC').sum()}")

    # Process files
    print(f"\n  Sorting files...")
    op     = "Moving" if use_move else "Copying"
    counts = {"PD": 0, "HC": 0, "unknown": 0}

    for nifti in nifti_files:
        # Determine class
        if args.label:
            cls = args.label
        elif metadata is not None:
            cls = match_subject(nifti, metadata)
        else:
            cls = None

        if cls not in CLASS_LABELS:
            counts["unknown"] += 1
            print(f"  ⚠️  Cannot determine class for: {nifti.name}")
            continue

        dest = Path(args.output_dir) / cls / nifti.name

        # Avoid overwriting
        if dest.exists():
            stem  = nifti.stem.replace(".nii", "")
            dest  = Path(args.output_dir) / cls / f"{stem}_dup{counts[cls]}.nii.gz"

        if use_move:
            shutil.move(str(nifti), str(dest))
        else:
            shutil.copy2(str(nifti), str(dest))

        counts[cls] += 1

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  Sort Complete")
    print(f"  {'─' * 45}")
    print(f"   PD     : {counts['PD']} files")
    print(f"   HC     : {counts['HC']} files")
    if counts["unknown"] > 0:
        print(f"   Unknown: {counts['unknown']} (no label found)")
    print(f"\n  Output: {os.path.abspath(args.output_dir)}")
    print(f"  Next  : run register.py → skull_strip.py → bias_correct.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()