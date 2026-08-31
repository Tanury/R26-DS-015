"""
check_mindglide_on_lemon_hc.py
================================
R26-DS-015 -- MRI-MS branch (MindGlide-based)

Runs MindGlide on LEMON healthy-control subjects and reports the lesion
volume it finds for each. Unlike MS3SEG, there's no ground-truth mask
here (these subjects are healthy) -- the question isn't "how accurate
is the segmentation" but "does MindGlide correctly find near-zero
lesion volume on brains that don't have MS lesions". This is the more
directly relevant test for whether MindGlide's volumes will actually
separate MS from Healthy in the eventual classifier.

File discovery works with ANY folder layout -- the full nested BIDS
structure (sub-X/ses-Y/anat/...) or a flat folder of files exactly as
downloaded from the browser, since subject ID and sequence type are
parsed directly from each filename (e.g. "sub-010001_ses-02_acq-lowres_
FLAIR.nii.gz" -> subject "sub-010001", sequence "FLAIR") rather than
inferred from directory structure. Searches recursively either way.

Usage:
    python3 check_mindglide_on_lemon_hc.py /path/to/LEMON --device mps --sw-batch-size 1
"""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from mindglide import segment

LESION_LABEL = 18


def parse_subject_id(filename: str) -> str | None:
    """Extract 'sub-010001' from a BIDS-style filename. Returns None if the
    filename doesn't start with a recognisable subject tag."""
    match = re.match(r"(sub-[a-zA-Z0-9]+)", filename)
    return match.group(1) if match else None


def discover_subject_files(root: Path) -> dict:
    """Recursively find every FLAIR/T1w NIfTI under root, regardless of
    folder layout, grouped by subject ID parsed from the filename itself.
    If a subject has multiple matching files (e.g. two sessions), the first
    one found (sorted by path for determinism) is kept."""
    subjects: dict = {}
    for pattern, seq_name in [("*_FLAIR.nii.gz", "FLAIR"), ("*_T1w.nii.gz", "T1w")]:
        for f in sorted(root.rglob(pattern)):
            sid = parse_subject_id(f.name)
            if sid is None:
                continue
            subjects.setdefault(sid, {})
            subjects[sid].setdefault(seq_name, f)
    return subjects


def process_subject(subject_id: str, files: dict, work_dir: Path, device: str, sw_batch_size: int) -> dict:
    if "FLAIR" in files:
        input_path = files["FLAIR"]
        sequence_used = "FLAIR"
    elif "T1w" in files:
        input_path = files["T1w"]
        sequence_used = "T1w"
    else:
        return {"subject_id": subject_id, "error": "No FLAIR or T1w file found for this subject"}

    seg_out_path = work_dir / f"{subject_id}_mindglide_seg.nii.gz"
    try:
        segment(str(input_path), str(seg_out_path), device=device, sw_batch_size=sw_batch_size)
    except Exception as e:
        return {"subject_id": subject_id, "error": f"{type(e).__name__}: {e}"}

    seg_img = nib.load(str(seg_out_path))
    seg_data = seg_img.get_fdata()
    lesion_mask = (seg_data == LESION_LABEL)

    zooms = seg_img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(np.prod(zooms))
    lesion_volume_mm3 = float(lesion_mask.sum() * voxel_vol_mm3)
    total_brain_voxels = float((seg_data > 0).sum())
    lesion_fraction = float(lesion_mask.sum() / total_brain_voxels) if total_brain_voxels > 0 else float("nan")

    return {
        "subject_id": subject_id,
        "sequence_used": sequence_used,
        "source_file": input_path.name,
        "lesion_volume_mm3": lesion_volume_mm3,
        "lesion_fraction_of_brain": lesion_fraction,
    }


def main(lemon_root: str, out_csv: str, device: str, sw_batch_size: int) -> None:
    root = Path(lemon_root)
    subjects = discover_subject_files(root)
    work_dir = Path("mindglide_lemon_work")
    work_dir.mkdir(exist_ok=True)

    print(f"Found {len(subjects)} subjects with usable files under {root} "
          f"(searched recursively, any folder layout)\n")

    results = []
    for subject_id in sorted(subjects.keys()):
        print(f"  Processing {subject_id}...")
        result = process_subject(subject_id, subjects[subject_id], work_dir, device, sw_batch_size)
        if "error" in result:
            print(f"    ⚠️  {result['error']}")
        else:
            print(f"    Lesion volume: {result['lesion_volume_mm3']:.0f} mm³  "
                  f"({result['lesion_fraction_of_brain']*100:.3f}% of brain, "
                  f"sequence: {result['sequence_used']}, file: {result['source_file']})")
        results.append(result)
        pd.DataFrame(results).to_csv(out_csv, index=False)

    df = pd.DataFrame(results)
    valid = df[~df.get("error", pd.Series(dtype=object)).notna()] if "error" in df.columns else df
    if len(valid) > 0 and "lesion_volume_mm3" in valid.columns:
        print("\n" + "=" * 60)
        print("SUMMARY -- LEMON healthy controls")
        print("=" * 60)
        print(f"Subjects successfully processed: {len(valid)}/{len(subjects)}")
        print(f"Mean lesion volume: {valid['lesion_volume_mm3'].mean():.0f} mm³ "
              f"(std: {valid['lesion_volume_mm3'].std():.0f})")
        print(f"Min / Max: {valid['lesion_volume_mm3'].min():.0f} / {valid['lesion_volume_mm3'].max():.0f} mm³")
        print("\nCompare this distribution against the MS3SEG patients' ground-truth lesion")
        print("volumes from the earlier validation run. For MindGlide's volumes to be a")
        print("usable classifier feature, this LEMON distribution should sit clearly below")
        print("(ideally near zero, and non-overlapping with) the MS3SEG patients' range.")
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lemon_root", help="Path to the folder containing LEMON files -- any layout, flat or nested")
    parser.add_argument("--out", default="mindglide_lemon_hc.csv")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--sw-batch-size", type=int, default=1)
    args = parser.parse_args()
    main(args.lemon_root, args.out, args.device, args.sw_batch_size)