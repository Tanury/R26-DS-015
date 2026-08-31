"""
Step 4 of the Alzheimer's MRI Preprocessing Pipeline

    Takes skull-stripped NIfTI files from data/skull_stripped/
    Runs N4 Bias Field Correction using ANTs (via antspyx Python library)
    Removes intensity non-uniformity caused by MRI scanner imperfections
    Supports batched processing for large datasets
    Preserves AD/MCI/CN subfolder structure in output

Runs sequentially (not parallel) because antspyx already uses all CPU cores internally per file. 
Parallelising on top causes thread contention on M-series Macs and slows things down.
"""

import os
import argparse
import math
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIGURATION
# All paths relative to R26-DS-015/
# ─────────────────────────────────────────────

DEFAULT_INPUT_DIR  = "mri/alzheimers/data/skull_stripped"
DEFAULT_OUTPUT_DIR = "mri/alzheimers/data/denoised"

# N4 parameters — standard settings for T1-weighted brain MRI
N4_ITERATIONS         = [50, 50, 50, 50]
N4_CONVERGENCE_THRESH = 0.0001
N4_SHRINK_FACTOR      = 4
N4_SPLINE_DISTANCE    = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="N4 Bias Field Correction using ANTs (antspyx)."
    )
    parser.add_argument("--input_dir",   type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir",  type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size",  type=int, default=None,
                        help="Files per batch. Omit to process all at once.")
    parser.add_argument("--batch_index", type=int, default=None,
                        help="0-indexed batch to process. Requires --batch_size.")
    return parser.parse_args()


def check_antspyx():
    try:
        import ants
        print(f"  antspyx v{ants.__version__} verified.")
        return ants
    except ImportError:
        print(f"\n antspyx not found.")
        print(f"    Install with: pip install antspyx")
        print(f"    Activate venv first: source .venv/bin/activate")
        raise


def discover_files(input_dir: str) -> list:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in sorted(filenames):
            if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                files.append(Path(root) / filename)
    return sorted(files)


def build_output_path(input_file: Path, input_dir: str, output_dir: str) -> Path:
    relative_path   = input_file.relative_to(input_dir)
    stem            = input_file.name.replace(".nii.gz", "").replace(".nii", "")
    output_filename = f"{stem}_n4.nii.gz"
    output_path     = Path(output_dir) / relative_path.parent / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def get_batch(files: list, batch_size: int, batch_index: int) -> tuple:
    total_batches = math.ceil(len(files) / batch_size)
    if batch_index >= total_batches:
        raise ValueError(
            f"batch_index {batch_index} out of range. "
            f"Valid indices: 0 to {total_batches - 1}."
        )
    start = batch_index * batch_size
    end   = min(start + batch_size, len(files))
    return files[start:end], total_batches


def correct_single_file(ants, input_file: Path, output_file: Path) -> dict:
    """
    Run N4 Bias Field Correction on one NIfTI file.
    Steps:
      1. Load the skull-stripped brain image
      2. Create a binary brain mask from non-zero voxels
      3. Run N4BiasFieldCorrection with the mask
      4. Save corrected image
    """
    file_start = datetime.now()

    try:
        image = ants.image_read(str(input_file))
        mask  = ants.get_mask(image, low_thresh=0.01, high_thresh=None, cleanup=2)

        corrected = ants.n4_bias_field_correction(
            image,
            mask              = mask,
            shrink_factor     = N4_SHRINK_FACTOR,
            convergence       = {
                "iters": N4_ITERATIONS,
                "tol":   N4_CONVERGENCE_THRESH
            },
            spline_param      = N4_SPLINE_DISTANCE,
            return_bias_field = False,
            verbose           = False
        )

        ants.image_write(corrected, str(output_file))

        return {
            "file":    input_file.name,
            "status":  "success",
            "elapsed": (datetime.now() - file_start).seconds,
            "error":   None
        }

    except Exception as e:
        return {
            "file":    input_file.name,
            "status":  "failed",
            "elapsed": (datetime.now() - file_start).seconds,
            "error":   str(e)
        }


def run_bias_correction(ants, files, input_dir, output_dir, batch_label=""):
    total   = len(files)
    success = 0
    skipped = 0
    failed  = []

    label_str = f" [{batch_label}]" if batch_label else ""
    print(f"\n  Processing {total} files sequentially{label_str}...")
    print(f"  N4 settings: iters={N4_ITERATIONS}, shrink={N4_SHRINK_FACTOR}, "
          f"spline={N4_SPLINE_DISTANCE}mm\n")

    run_start = datetime.now()

    for i, input_file in enumerate(files, 1):
        output_file = build_output_path(input_file, input_dir, output_dir)

        if output_file.exists():
            skipped += 1
            continue

        result = correct_single_file(ants, input_file, output_file)

        if result["status"] == "success":
            success += 1
            elapsed_total = (datetime.now() - run_start).seconds
            print(
                f"  [{i}/{total}]  {result['file']}"
                f"  (file: {result['elapsed']}s | total: "
                f"{elapsed_total // 60}m {elapsed_total % 60}s)"
            )
        else:
            failed.append({"file": result["file"], "error": result["error"]})
            print(f"  [{i}/{total}]  {result['file']}")
            print(f"              {result['error']}")

        # Estimated time remaining every 10 files
        if i % 10 == 0 and success > 0:
            elapsed_total = (datetime.now() - run_start).seconds
            avg           = elapsed_total / (success + len(failed))
            remaining     = avg * (total - i)
            print(
                f"\n  ⏳  {i}/{total} done. "
                f"Est. remaining: {int(remaining // 60)}m {int(remaining % 60)}s\n"
            )

    return success, skipped, failed, run_start


def print_summary(success, skipped, failed, output_dir, start_time,
                  batch_size=None, batch_index=None, total_batches=None):
    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'=' * 55}")
    print(f"  Bias Correction Complete")
    print(f"  {'─' * 45}")
    print(f"   Corrected : {success}")
    print(f"   Skipped   : {skipped}")
    print(f"   Failed    : {len(failed)}")
    if failed:
        for item in failed:
            print(f"    - {item['file']}: {item['error']}")

    if batch_size and total_batches and batch_index is not None:
        next_batch = batch_index + 1
        if next_batch < total_batches:
            print(f"\n  📦  Batch {batch_index + 1}/{total_batches} done.")
            print(f"      Run next batch:")
            print(f"      python mri/alzheimers/preprocessing/bias_correct.py "
                  f"--batch_size {batch_size} --batch_index {next_batch}")
        else:
            print(f"\n  🎉  All {total_batches} batches complete!")

    print(f"\n    Time: {elapsed // 60}m {elapsed % 60}s")
    print(f"   Output: {os.path.abspath(output_dir)}")
    print(f"\n   Preprocessing complete!")
    print(f"      Files in data/denoised/ are ready for train.py\n")
    print(f"{'=' * 55}\n")


def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Alzheimer's MRI — Step 4 of 4: Bias Correction")
    print("=" * 55)

    print("\n[1/4] Checking environment...")
    ants = check_antspyx()

    print(f"\n[2/4] Scanning: {args.input_dir}")
    all_files = discover_files(args.input_dir)
    if not all_files:
        print(f"\n  No NIfTI files found. Run skull_strip.py first.")
        return
    print(f"  Found {len(all_files)} files.")

    print(f"\n[3/4] Preparing output dirs...")
    for label in ["AD", "MCI", "CN"]:
        Path(args.output_dir, label).mkdir(parents=True, exist_ok=True)

    # Handle batching
    files_to_process = all_files
    total_batches    = None
    batch_label      = ""

    if args.batch_size is not None:
        batch_index      = args.batch_index if args.batch_index is not None else 0
        files_to_process, total_batches = get_batch(
            all_files, args.batch_size, batch_index
        )
        batch_label = f"Batch {batch_index + 1}/{total_batches}"
        print(f"\n    {batch_label}: "
              f"files {batch_index * args.batch_size + 1}–"
              f"{batch_index * args.batch_size + len(files_to_process)} "
              f"of {len(all_files)}")
    elif args.batch_index is not None:
        print(f"\n  ⚠️  --batch_index given without --batch_size. Processing all files.")

    print(f"\n[4/4] Starting N4 bias correction...")
    print(f"     ~30–120 seconds per file on M4.")
    print(f"    CPU will be fully used — some warmth is normal.")

    success, skipped, failed, start_time = run_bias_correction(
        ants, files_to_process, args.input_dir, args.output_dir, batch_label
    )

    print_summary(
        success, skipped, failed, args.output_dir, start_time,
        args.batch_size, args.batch_index, total_batches
    )


if __name__ == "__main__":
    main()