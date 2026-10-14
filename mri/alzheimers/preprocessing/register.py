"""
register.py -  Step 2 of Alzheimer's MRI Preprocessing Pipeline
-----------------------------------------------------
it :
Takes sorted NIfTI files from output of compile.py ==> data/sorted/
Runs FSL fslreorient2std : reorients to MNI152 standard orientation
Runs FSL flirt : affine registration to MNI152 atlas
Uses multiprocessing to process multiple files in parallel
Preserves AD/MCI/CN subfolder structure in output

    Input  → mri/alzheimers/data/sorted/{AD,MCI,CN}/*.nii.gz
    Output → mri/alzheimers/data/registered/{AD,MCI,CN}/*_reg.nii.gz
"""

import os
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime

DEFAULT_INPUT_DIR  = "mri/alzheimers/data/sorted"
DEFAULT_OUTPUT_DIR = "mri/alzheimers/data/registered"
DEFAULT_ATLAS_PATH = "shared/atlas/MNI152_T1_1mm.nii.gz"

FLIRT_DOF    = 12
FLIRT_INTERP = "trilinear"
FLIRT_COST   = "corratio"

DEFAULT_CORES = None  # None = all available cores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Affine registration of ADNI NIfTI files using FSL."
    )
    parser.add_argument("--input_dir",  type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--atlas",      type=str, default=DEFAULT_ATLAS_PATH)
    parser.add_argument("--cores",      type=int, default=DEFAULT_CORES)
    return parser.parse_args()


def check_fsl():
    for tool in ["flirt", "fslreorient2std"]:
        result = subprocess.run(["which", tool], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n FSL tool '{tool}' not found on PATH.")
            print(f"    Add FSL to ~/.zshrc:")
            print(f"      export FSLDIR=/Users/tanuridissanayaka/fsl")
            print(f"      source $FSLDIR/etc/fslconf/fsl.sh")
            print(f"      export PATH=$FSLDIR/bin:$PATH")
            print(f"    Then: source ~/.zshrc")
            raise EnvironmentError(f"FSL tool '{tool}' not on PATH.")
    print("  FSL verified (flirt, fslreorient2std)")


def check_atlas(atlas_path: str):
    if not os.path.exists(atlas_path):
        print(f"\n  Atlas not found at: {atlas_path}")
        print(f"    Copy it with:")
        print(f"      cp /Users/tanuridissanayaka/fsl/data/standard/MNI152_T1_1mm.nii shared/atlas/")
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")
    print(f"  Atlas found: {atlas_path}")


def discover_files(input_dir: str) -> list:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                files.append(Path(root) / filename)
    return files


def build_output_path(input_file: Path, input_dir: str, output_dir: str) -> Path:
    relative_path   = input_file.relative_to(input_dir)
    stem            = input_file.name.replace(".nii.gz", "").replace(".nii", "")
    output_filename = f"{stem}_reg.nii.gz"
    output_path     = Path(output_dir) / relative_path.parent / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def reorient_file(input_file: Path, temp_dir: Path) -> Path:
    stem          = input_file.name.replace(".nii.gz", "").replace(".nii", "")
    reoriented    = temp_dir / f"{stem}_reoriented.nii.gz"

    result = subprocess.run(
        ["fslreorient2std", str(input_file), str(reoriented)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"fslreorient2std failed: {result.stderr}")
    return reoriented


def register_file(reoriented_file: Path, output_file: Path, atlas_path: str):
    cmd = [
        "flirt",
        "-in",     str(reoriented_file),
        "-ref",    atlas_path,
        "-out",    str(output_file),
        "-dof",    str(FLIRT_DOF),
        "-interp", FLIRT_INTERP,
        "-cost",   FLIRT_COST,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"flirt failed: {result.stderr}")


def process_single_file(args_tuple: tuple) -> dict:
    input_file, input_dir, output_dir, atlas_path, temp_dir = args_tuple
    output_file = build_output_path(input_file, input_dir, output_dir)

    if output_file.exists():
        return {"file": input_file.name, "status": "skipped", "error": None}

    try:
        reoriented = reorient_file(input_file, Path(temp_dir))
        register_file(reoriented, output_file, atlas_path)
        if reoriented.exists():
            reoriented.unlink()
        return {"file": input_file.name, "status": "success", "error": None}

    except Exception as e:
        return {"file": input_file.name, "status": "failed", "error": str(e)}


def run_registration(files, input_dir, output_dir, atlas_path, n_cores):
    temp_dir = Path(output_dir) / "_temp_reoriented"
    temp_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (f, input_dir, output_dir, atlas_path, str(temp_dir))
        for f in files
    ]

    total   = len(worker_args)
    success = 0
    skipped = 0
    failed  = []

    print(f"\n  Processing {total} files using {n_cores} core(s)...")
    print(f"  This is the slowest step — affine registration is compute-intensive.\n")

    start_time = datetime.now()

    with multiprocessing.Pool(processes=n_cores) as pool:
        for i, result in enumerate(
            pool.imap_unordered(process_single_file, worker_args), 1
        ):
            status = result["status"]
            fname  = result["file"]

            if status == "success":
                success += 1
                if i % 10 == 0 or i == total:
                    elapsed = (datetime.now() - start_time).seconds
                    print(f"  [{i}/{total}] {fname}  ({elapsed}s elapsed)")
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed.append(fname)
                print(f"  [{i}/{total}] {fname}")
                print(f"              {result['error']}")

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    return success, skipped, failed


def print_summary(success, skipped, failed, output_dir, start_time):
    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'=' * 55}")
    print(f"  Registration Complete")
    print(f"  {'─' * 45}")
    print(f"  Registered : {success}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        for f in failed:
            print(f"    - {f}")
    print(f"\n  Time: {elapsed // 60}m {elapsed % 60}s")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"\n  Next step: run skull_strip.py\n")
    print(f"{'=' * 55}\n")


def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Alzheimer's MRI — Step 2 of 4: Register")
    print("=" * 55)

    print("\n[1/4] Checking environment...")
    check_fsl()
    check_atlas(args.atlas)

    print(f"\n[2/4] Scanning: {args.input_dir}")
    files = discover_files(args.input_dir)
    if not files:
        print(f"\n No NIfTI files found. Run compile.py first.")
        return
    print(f"  Found {len(files)} files.")

    print(f"\n[3/4] Preparing output dirs...")
    for label in ["AD", "MCI", "CN"]:
        Path(args.output_dir, label).mkdir(parents=True, exist_ok=True)

    available = multiprocessing.cpu_count()
    n_cores   = args.cores if args.cores else available
    print(f"\n[4/4] Starting registration ({n_cores}/{available} cores)...")

    start_time = datetime.now()
    success, skipped, failed = run_registration(
        files, args.input_dir, args.output_dir, args.atlas, n_cores
    )
    print_summary(success, skipped, failed, args.output_dir, start_time)


if __name__ == "__main__":
    main()