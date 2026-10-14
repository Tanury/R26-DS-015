"""
Step 0 of Alzheimer's MRI Preprocessing Pipeline

    For each scan (leaf folder containing .dcm files):
        - Runs dcm2niix to convert DICOM → NIfTI
        - Extracts the Image ID from the folder path (e.g. I254580)
        - Renames output to ADNI_<SubjectID>_I<ImageID>.nii.gz
        - Places it flat into data/raw/

    Output → mri/alzheimers/data/raw/ADNI_<SubjectID>_I<ImageID>.nii.gz

Usage:
    python mri/alzheimers/preprocessing/dcm2nifti_convert.py \\
        --dicom_dir "/Users/tanuridissanayaka/Desktop/research/ADNI 2" \\
        --output_dir "mri/alzheimers/data/raw"

"""

import os
import re
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


DEFAULT_DICOM_DIR  = "/Users/tanuridissanayaka/Desktop/research/ADNI 2"
DEFAULT_OUTPUT_DIR = "mri/alzheimers/data/raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ADNI DICOM folders to flat NIfTI files for the preprocessing pipeline."
    )
    parser.add_argument("--dicom_dir",  type=str, default=DEFAULT_DICOM_DIR,
                        help="Root folder of your ADNI DICOM download.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Where to place converted .nii.gz files.")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-convert files that already exist in output_dir.")
    return parser.parse_args()


def check_dcm2niix():
    result = subprocess.run(["which", "dcm2niix"], capture_output=True, text=True)
    if result.returncode != 0:
        print("\n  dcm2niix not found on PATH.")
        print("    Install with:")
        print("      brew install dcm2niix")
        print("    or download from: https://github.com/rordenlab/dcm2niix/releases")
        raise EnvironmentError("dcm2niix not on PATH.")
    result2 = subprocess.run(["dcm2niix", "--version"], capture_output=True, text=True)
    version = (result2.stdout + result2.stderr).strip().splitlines()[0]
    print(f"  dcm2niix found: {version}")


def find_dicom_series(dicom_root: str) -> list:
    """
    Walk the ADNI folder tree and find all leaf directories that contain .dcm files.
    Returns list of dicts: {dicom_dir, subject_id, image_id}

    ADNI structure:
        <root>/<SubjectID>/<ScanType>/<Date>/<ImageID>/*.dcm
    The ImageID folder name is a plain number like 35475.
    We prefix it with "I" to match ADNIMERGE IMAGEUID format → I35475.
    """
    series_list = []
    dicom_root_path = Path(dicom_root)

    for dirpath, dirnames, filenames in os.walk(dicom_root):
        dcm_files = [f for f in filenames if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        # This is a leaf DICOM folder
        parts = Path(dirpath).relative_to(dicom_root_path).parts

        # Expected: SubjectID / ScanType / Date / ImageID
        if len(parts) < 4:
            # Fallback: try to extract image ID from last numeric folder
            subject_id = parts[0] if len(parts) >= 1 else "UNKNOWN"
            image_id   = _extract_image_id_from_path(dirpath)
        else:
            subject_id = parts[0]   # e.g. "002_S_0295"
            image_id   = _extract_image_id_from_path(parts[3])  # e.g. "I35475"

        series_list.append({
            "dicom_dir":  dirpath,
            "subject_id": subject_id,
            "image_id":   image_id,
        })

    return series_list


def _extract_image_id_from_path(folder_name: str) -> str:
    """
    Extract numeric image ID from folder name and prefix with 'I'.
    ADNI image ID folders are plain numbers: e.g. '35475' → 'I35475'
    If the folder already starts with I followed by digits, use as-is.
    """
    folder_name = str(folder_name)
    # Already in correct format
    if re.match(r'^I\d+$', folder_name):
        return folder_name
    # Plain number
    match = re.search(r'(\d{4,})', folder_name)
    if match:
        return f"I{match.group(1)}"
    return f"I{folder_name}"


def convert_series(series: dict, output_dir: str, overwrite: bool) -> dict:
    """
    Convert one DICOM series to NIfTI using dcm2niix.
    Output filename: ADNI_<SubjectID>_<ImageID>.nii.gz
    """
    subject_id  = series["subject_id"]
    image_id    = series["image_id"]
    dicom_dir   = series["dicom_dir"]
    output_name = f"ADNI_{subject_id}_{image_id}"
    output_path = Path(output_dir) / f"{output_name}.nii.gz"

    if output_path.exists() and not overwrite:
        return {"status": "skipped", "output": str(output_path), "error": None}

    # Use a temp dir so dcm2niix output is isolated before we rename
    temp_dir = Path(output_dir) / "_dcm2niix_temp" / image_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "dcm2niix",
            "-z", "y",          # compress to .nii.gz
            "-f", output_name,  # filename pattern
            "-o", str(temp_dir),
            "-b", "n",          # skip BIDS sidecar .json (not needed for pipeline)
            str(dicom_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        # Find the .nii.gz dcm2niix created (it may append suffixes)
        nifti_files = list(temp_dir.glob("*.nii.gz"))
        if not nifti_files:
            nifti_files = list(temp_dir.glob("*.nii"))
        if not nifti_files:
            raise RuntimeError(
                "dcm2niix exited 0 but produced no NIfTI file.\n"
                f"  stdout: {result.stdout[-300:]}"
            )

        # If multiple files produced (multi-echo etc.), take the first
        src = nifti_files[0]

        # Rename to our standard pattern and move to output_dir
        final_name = output_name
        if not src.name.startswith(output_name):
            # dcm2niix added a suffix — keep it for uniqueness
            final_name = src.stem.replace(".nii", "")

        final_path = Path(output_dir) / f"{final_name}.nii.gz"
        shutil.move(str(src), str(final_path))

        return {"status": "success", "output": str(final_path), "error": None}

    except Exception as e:
        return {"status": "failed", "output": None, "error": str(e)}

    finally:
        # Clean up temp folder
        shutil.rmtree(str(temp_dir), ignore_errors=True)


def print_summary(success, skipped, failed, output_dir, start_time):
    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'=' * 55}")
    print(f"  DICOM → NIfTI Conversion Complete")
    print(f"  {'─' * 45}")
    print(f"   Converted : {success}")
    print(f"   Skipped   : {skipped}  (already exist — use --overwrite to redo)")
    print(f"   Failed    : {len(failed)}")
    if failed:
        print(f"\n  Failed series:")
        for f in failed[:10]:
            print(f"    - {f['dicom_dir']}")
            print(f"      {f['error']}")
    print(f"\n  ⏱️   Time   : {elapsed // 60}m {elapsed % 60}s")
    print(f"  📁  Output : {os.path.abspath(output_dir)}")
    print(f"\n  ✅  Next step: run compile.py\n")
    print(f"{'=' * 55}\n")


def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Alzheimer's MRI — Step 0: DICOM → NIfTI Conversion")
    print("=" * 55)

    print("\n[1/4] Checking environment...")
    check_dcm2niix()

    print(f"\n[2/4] Scanning DICOM root: {args.dicom_dir}")
    if not os.path.exists(args.dicom_dir):
        print(f"\n  Directory not found: {args.dicom_dir}")
        print(f"  Pass your ADNI download folder with --dicom_dir")
        return

    series_list = find_dicom_series(args.dicom_dir)
    if not series_list:
        print(f"\n  No DICOM series found under: {args.dicom_dir}")
        print(f"  Make sure the folder contains subdirectories with .dcm files.")
        return
    print(f"  Found {len(series_list)} DICOM series.")

    print(f"\n[3/4] Preparing output directory: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Converting...")
    if args.overwrite:
        print(f"  --overwrite enabled: re-converting existing files.")

    success = 0
    skipped = 0
    failed  = []
    total   = len(series_list)
    start_time = datetime.now()

    for i, series in enumerate(series_list, 1):
        label = f"{series['subject_id']} / {series['image_id']}"
        result = convert_series(series, args.output_dir, args.overwrite)

        if result["status"] == "success":
            success += 1
            if i % 10 == 0 or i == total:
                elapsed = (datetime.now() - start_time).seconds
                print(f"  [{i}/{total}]  {label}  ({elapsed}s elapsed)")
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed.append({**series, "error": result["error"]})
            print(f"  [{i}/{total}]  FAILED  {label}")
            print(f"               {result['error']}")

    print_summary(success, skipped, failed, args.output_dir, start_time)


if __name__ == "__main__":
    main()