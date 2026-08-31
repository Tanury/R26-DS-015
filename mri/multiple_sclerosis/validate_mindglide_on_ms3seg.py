"""
validate_mindglide_on_ms3seg.py

Validates MindGlide's automatic lesion segmentation against MS3SEG's
expert-annotated ground truth, on real MS patients, BEFORE trusting
MindGlide's output for the downstream classifier pipeline.

For each patient:
    1. Run MindGlide on the FLAIR volume (masks are in FLAIR space per
       MS3SEG's own co-registration pipeline).
    2. Extract MindGlide's "Lesion" label (code 18) as a binary mask.
    3. Load MS3SEG's expert abnormal-WMH ("abWMH") mask.
    4. Compute Dice overlap + compare lesion volumes (mm^3).

Usage:
    python3 validate_mindglide_on_ms3seg.py \\
        /path/to/MS3SEG_Dataset/MS_100_patient_preprocessed \\
        /path/to/MS3SEG_Dataset/MS_100_patient_masks/abWMH_Masks \\
        --n 10 --out mindglide_validation.csv

Notes:
    - Requires: pip install mindglide nibabel numpy pandas
    - MindGlide auto-downloads its model checkpoint (~123MB) from
      Hugging Face on first run -- needs normal internet access.
    - Runs on CPU by default if no GPU is available; a handful of
      patients (--n 10) is enough for a first validation pass before
      committing to segmenting all 100.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from mindglide import segment

LESION_LABEL = 18  # per MindGlide's label table (mindglide --labels)


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0  # both empty -- perfect agreement (no lesion, correctly found none)
    return 2.0 * intersection / denom


def voxel_volume_mm3(nifti_img) -> float:
    """Volume of a single voxel in mm^3, from the NIfTI header's voxel sizes."""
    zooms = nifti_img.header.get_zooms()[:3]
    return float(np.prod(zooms))


def validate_patient(
    flair_dir: Path, mask_dir: Path, patient_id: str, work_dir: Path,
    device: str, sw_batch_size: int
) -> dict:
    flair_path = flair_dir / patient_id / f"{patient_id}_FLAIR.nii.gz"
    mask_path = mask_dir / patient_id / f"{patient_id}_abWMH_Mask.nii.gz"

    if not flair_path.exists():
        return {"patient_id": patient_id, "error": f"FLAIR not found: {flair_path}"}
    if not mask_path.exists():
        return {"patient_id": patient_id, "error": f"Mask not found: {mask_path}"}

    seg_out_path = work_dir / f"{patient_id}_mindglide_seg.nii.gz"
    segment(str(flair_path), str(seg_out_path), device=device, sw_batch_size=sw_batch_size)

    seg_img = nib.load(str(seg_out_path))
    seg_data = seg_img.get_fdata()
    mindglide_lesion_mask = (seg_data == LESION_LABEL)

    gt_img = nib.load(str(mask_path))
    gt_mask = gt_img.get_fdata() > 0.5  # binarize, in case of float/probabilistic values

    if mindglide_lesion_mask.shape != gt_mask.shape:
        return {
            "patient_id": patient_id,
            "error": f"Shape mismatch: MindGlide seg {mindglide_lesion_mask.shape} "
                     f"vs ground truth {gt_mask.shape} -- likely a resampling/space issue, "
                     f"needs investigation before trusting the comparison.",
        }

    dice = dice_score(mindglide_lesion_mask, gt_mask)
    vox_vol = voxel_volume_mm3(seg_img)
    mindglide_volume_mm3 = mindglide_lesion_mask.sum() * vox_vol
    gt_volume_mm3 = gt_mask.sum() * vox_vol

    return {
        "patient_id": patient_id,
        "dice": dice,
        "mindglide_lesion_volume_mm3": mindglide_volume_mm3,
        "ground_truth_lesion_volume_mm3": gt_volume_mm3,
        "volume_ratio": (mindglide_volume_mm3 / gt_volume_mm3) if gt_volume_mm3 > 0 else float("nan"),
    }


def main(flair_dir: str, mask_dir: str, n: int, out_csv: str, device: str, sw_batch_size: int, restart: bool) -> None:
    flair_dir_p = Path(flair_dir)
    mask_dir_p = Path(mask_dir)
    work_dir = Path("mindglide_validation_work")
    work_dir.mkdir(exist_ok=True)

    patient_ids = sorted(p.name for p in flair_dir_p.iterdir() if p.is_dir())[:n]

    results = []
    already_done = set()
    out_path = Path(out_csv)
    if out_path.exists() and not restart:
        existing = pd.read_csv(out_path, dtype={"patient_id": str})
        results = existing.to_dict("records")
        already_done = set(existing["patient_id"])
        print(f"Found existing {out_csv} with {len(already_done)} patients already processed -- resuming.")
        print(f"(Use --restart to ignore this and start over from scratch.)\n")

    remaining = [pid for pid in patient_ids if pid not in already_done]
    print(f"Validating MindGlide on {len(patient_ids)} MS3SEG patients "
          f"({len(already_done)} already done, {len(remaining)} remaining)\n")

    for pid in remaining:
        print(f"  Processing {pid}...")
        try:
            result = validate_patient(flair_dir_p, mask_dir_p, pid, work_dir, device, sw_batch_size)
        except Exception as e:
            result = {"patient_id": pid, "error": f"{type(e).__name__}: {e}"}

        if "error" in result:
            print(f"    ⚠️  {result['error']}")
        else:
            print(f"    Dice: {result['dice']:.3f}  |  "
                  f"MindGlide vol: {result['mindglide_lesion_volume_mm3']:.0f} mm³  |  "
                  f"Ground truth vol: {result['ground_truth_lesion_volume_mm3']:.0f} mm³")
        results.append(result)
        pd.DataFrame(results).to_csv(out_csv, index=False)

    df = pd.DataFrame(results)
    valid = df[~df.get("error", pd.Series(dtype=object)).notna()] if "error" in df.columns else df
    if len(valid) > 0 and "dice" in valid.columns:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Patients successfully processed: {len(valid)}/{len(patient_ids)}")
        print(f"Mean Dice: {valid['dice'].mean():.3f}  (std: {valid['dice'].std():.3f})")
        print(f"Mean volume ratio (MindGlide/ground truth): {valid['volume_ratio'].mean():.3f}")
        print("\nInterpretation guide:")
        print("  Dice > 0.6  -- generally considered good agreement for lesion segmentation")
        print("  Dice 0.4-0.6 -- moderate; MindGlide's volumes may still rank-order subjects")
        print("                  usefully even if pixel-level overlap isn't perfect")
        print("  Dice < 0.4  -- weak agreement; investigate before trusting MindGlide's")
        print("                  lesion volumes as a classifier feature on this cohort")
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("flair_dir", help="Path to MS_100_patient_preprocessed (or similar, containing <id>/<id>_FLAIR.nii.gz)")
    parser.add_argument("mask_dir", help="Path to MS_100_patient_masks/abWMH_Masks")
    parser.add_argument("--n", type=int, default=10, help="Number of patients to validate (default 10)")
    parser.add_argument("--out", default="mindglide_validation.csv")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda", "auto"],
                         help="mps (Apple GPU) is much faster than cpu but can run out of memory on "
                              "repeated large 3D volumes -- if it crashes, lower --sw-batch-size first "
                              "before falling back to cpu, which is dramatically slower on some machines.")
    parser.add_argument("--sw-batch-size", type=int, default=1,
                         help="MindGlide's internal sliding-window batch size (default library value is "
                              "4). Lower = less memory per step, at some cost to speed. Start at 1 if "
                              "you hit an MPS/CUDA out-of-memory error.")
    parser.add_argument("--restart", action="store_true",
                         help="Ignore any existing --out CSV and start over from scratch. Without this "
                              "flag, if --out already exists, already-processed patients are skipped "
                              "and only the remaining ones are run.")
    args = parser.parse_args()
    main(args.flair_dir, args.mask_dir, args.n, args.out, args.device, args.sw_batch_size, args.restart)