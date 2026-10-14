"""
skull_strip.py - Step 3 of Alzheimer MRI Preprocessing Pipeline

What this does:
     Takes registered NIfTI files from data/registered/ (output of register.py)
     Runs FSL BET (Brain Extraction Tool) to remove skull, eyes, and other non-brain tissue from each scan
     Uses multiprocessing to process multiple files in parallel
     Preserves AD/MCI/CN subfolder structure in output

    Input  → mri/alzheimers/data/registered/{AD,MCI,CN}/*_reg.nii.gz
    Output → mri/alzheimers/data/skull_stripped/{AD,MCI,CN}/*_brain.nii.gz


BET Fractional Intensity Threshold (--frac):
    Range: 0.0 (most aggressive) to 1.0 (least aggressive). Default: 0.5
    Too much skull left → lower value (try 0.3)
    Brain tissue removed → raise value (try 0.7)

"""

import os
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime


DEFAULT_INPUT_DIR  = "mri/alzheimers/data/registered"
DEFAULT_OUTPUT_DIR = "mri/alzheimers/data/skull_stripped"

DEFAULT_FRAC    = 0.5
BET_EXTRA_FLAGS = ["-R"]
DEFAULT_CORES   = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Skull stripping of registered ADNI NIfTI files using FSL BET."
    )
    parser.add_argument("--input_dir",  type=str,  default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str,  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frac",       type=float, default=DEFAULT_FRAC)
    parser.add_argument("--cores",      type=int,  default=DEFAULT_CORES)
    return parser.parse_args()


def check_fsl():
    result = subprocess.run(["which", "bet"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n FSL 'bet' not found on PATH.")
        print(f"    Add to ~/.zshrc:")
        print(f"      export FSLDIR=/Users/tanuridissanayaka/fsl")
        print(f"      source $FSLDIR/etc/fslconf/fsl.sh")
        print(f"      export PATH=$FSLDIR/bin:$PATH")
        raise EnvironmentError("FSL 'bet' not on PATH.")
    print(f"  FSL BET verified.")


def discover_files(input_dir: str) -> list:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if (filename.endswith(".nii.gz") or filename.endswith(".nii")):
                if "_brain" not in filename:
                    files.append(Path(root) / filename)
    return files


def build_output_path(input_file: Path, input_dir: str, output_dir: str) -> Path:
    relative_path   = input_file.relative_to(input_dir)
    stem            = input_file.name.replace(".nii.gz", "").replace(".nii", "")
    output_filename = f"{stem}_brain.nii.gz"
    output_path     = Path(output_dir) / relative_path.parent / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def strip_single_file(args_tuple: tuple) -> dict:
    input_file, input_dir, output_dir, frac = args_tuple
    output_file = build_output_path(input_file, input_dir, output_dir)

    if output_file.exists():
        return {"file": input_file.name, "status": "skipped", "error": None}

    try:
        cmd = (
            ["bet", str(input_file), str(output_file)]
            + ["-f", str(frac)]
            + BET_EXTRA_FLAGS
        )
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        if not output_file.exists():
            raise RuntimeError(
                "BET exited 0 but no output file created. "
                "Try adjusting --frac (e.g. --frac 0.4)."
            )

        return {"file": input_file.name, "status": "success", "error": None}

    except Exception as e:
        return {"file": input_file.name, "status": "failed", "error": str(e)}


def run_skull_stripping(files, input_dir, output_dir, frac, n_cores):
    worker_args = [(f, input_dir, output_dir, frac) for f in files]

    total   = len(worker_args)
    success = 0
    skipped = 0
    failed  = []

    print(f"\n  Processing {total} files using {n_cores} core(s)...")
    print(f"  BET threshold: {frac}  |  Extra flags: {' '.join(BET_EXTRA_FLAGS)}\n")

    start_time = datetime.now()

    with multiprocessing.Pool(processes=n_cores) as pool:
        for i, result in enumerate(
            pool.imap_unordered(strip_single_file, worker_args), 1
        ):
            status = result["status"]
            fname  = result["file"]

            if status == "success":
                success += 1
                if i % 10 == 0 or i == total:
                    elapsed = (datetime.now() - start_time).seconds
                    print(f"  [{i}/{total}]   {fname}  ({elapsed}s elapsed)")
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed.append({"file": fname, "error": result["error"]})
                print(f"  [{i}/{total}]  {fname}")
                print(f"              {result['error']}")

    return success, skipped, failed, start_time


def print_summary(success, skipped, failed, output_dir, start_time, frac):
    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'=' * 55}")
    print(f"  Skull Stripping Complete")
    print(f"  {'─' * 45}")
    print(f"   Stripped : {success}")
    print(f"   Skipped  : {skipped}")
    print(f"   Failed   : {len(failed)}")
    if failed:
        for item in failed:
            print(f"    - {item['file']}: {item['error']}")
        print(f"\n  💡  Tip: Try adjusting --frac (current: {frac})")
        print(f"      Less aggressive: --frac 0.4 or --frac 0.3")
    print(f"\n  ⏱️   Time: {elapsed // 60}m {elapsed % 60}s")
    print(f"  📁  Output: {os.path.abspath(output_dir)}")
    print(f"\n   Next step: run bias_correct.py\n")
    print(f"{'=' * 55}\n")


def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Alzheimer's MRI — Step 3 of 4: Skull Strip")
    print("=" * 55)

    print("\n[1/4] Checking environment...")
    check_fsl()

    print(f"\n[2/4] Scanning: {args.input_dir}")
    files = discover_files(args.input_dir)
    if not files:
        print(f"\n  No NIfTI files found. Run register.py first.")
        return
    print(f"  Found {len(files)} files.")

    print(f"\n[3/4] Preparing output dirs...")
    for label in ["AD", "MCI", "CN"]:
        Path(args.output_dir, label).mkdir(parents=True, exist_ok=True)

    available = multiprocessing.cpu_count()
    n_cores   = args.cores if args.cores else available
    print(f"\n[4/4] Starting skull stripping ({n_cores}/{available} cores)...")

    success, skipped, failed, start_time = run_skull_stripping(
        files, args.input_dir, args.output_dir, args.frac, n_cores
    )
    print_summary(success, skipped, failed, args.output_dir, start_time, args.frac)


if __name__ == "__main__":
    main()