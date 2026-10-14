"""
embed.py
========
Batch z_img Embedding Extractor — Alzheimer's MRI
--------------------------------------------------
What this script does:
    - Loads the trained VisionEncoder from best_encoder.pth
    - Runs inference on ALL preprocessed files in data/denoised/
    - Extracts the 256-d L2-normalized z_img embedding for every scan
    - Saves results to outputs/alzheimers_mri/z_img_all.npz

    This is the primary handoff script between your component and
    your team's adaptive fusion engine (Aarabhi A.Y.).

    The .npz file contains:
        embeddings : (N, 256) float32 — the z_img vectors
        labels     : (N,)    int      — class label per file
        paths      : (N,)    str      — source file path per embedding
        class_names: list    str      — ["AD", "MCI", "CN"]
        severity   : (N,)    int      — severity label (CN=0, MCI=1, AD=2)

Run from project root (R26-DS-015/):
    python mri/alzheimers/models/embed.py

    Custom data directory:
    python mri/alzheimers/models/embed.py \
        --data_dir mri/alzheimers/data/denoised \
        --output_dir outputs/alzheimers_mri

Author: R26-DS-015 Vision Encoder
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    Compose, EnsureChannelFirst, Resize, ScaleIntensity,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mri.alzheimers.models.vision_encoder import build_encoder


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_DATA_DIR       = "mri/alzheimers/data/denoised"
DEFAULT_CHECKPOINT_DIR = "mri/alzheimers/models/checkpoints"
DEFAULT_OUTPUT_DIR     = "outputs/alzheimers_mri"
DEFAULT_IMAGE_SIZE     = (96, 96, 96)
DEFAULT_BATCH_SIZE     = 2

CLASS_LABELS       = ["AD", "MCI", "CN"]
CLASS_TO_SEVERITY  = {0: 2, 1: 1, 2: 0}   # AD=2, MCI=1, CN=0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract z_img embeddings from all preprocessed MRI files."
    )
    parser.add_argument("--data_dir",       type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output_dir",     type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size",     type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 58)
    print("  Alzheimer's MRI — z_img Embedding Extraction")
    print("  Output: 256-d L2-normalized embeddings for fusion engine")
    print("=" * 58)

    # Device
    device = get_device()
    print(f"\n  Device: {device}")

    # Load encoder
    ckpt = Path(args.checkpoint_dir) / "best_model.pth"
    if not ckpt.exists():
        print(f"\n❌  Checkpoint not found: {ckpt}")
        print(f"    Run train.py first.")
        return

    print(f"\n  Loading encoder from: {ckpt}")
    model = build_encoder(num_classes=len(CLASS_LABELS), device=device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()
    print(f"  ✅  Encoder loaded.")

    # Discover all files
    all_paths  = []
    all_labels = []
    print(f"\n  Scanning: {args.data_dir}")

    for label_idx, label_name in enumerate(CLASS_LABELS):
        class_dir = Path(args.data_dir) / label_name
        if not class_dir.exists():
            print(f"    {label_name}: ⚠️  not found")
            continue
        files = sorted([
            f for f in class_dir.iterdir()
            if f.name.endswith(".nii.gz") or f.name.endswith(".nii")
        ])
        for f in files:
            all_paths.append(str(f))
            all_labels.append(label_idx)
        print(f"    {label_name}: {len(files)} files")

    if not all_paths:
        print(f"\n❌  No files found in {args.data_dir}")
        return

    print(f"\n  Total: {len(all_paths)} files → extracting 256-d z_img per file...")

    # Transforms
    transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(DEFAULT_IMAGE_SIZE),
    ])

    # DataLoader
    ds     = ImageDataset(image_files=all_paths, labels=all_labels, transform=transforms)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Extract embeddings
    all_embeds   = []
    total        = len(all_paths)
    processed    = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            z_img  = model.encode(inputs)   # (batch, 256)
            all_embeds.extend(z_img.cpu().numpy())
            processed += len(inputs)
            print(f"  [{processed:>4}/{total}] embeddings extracted...")

    embeddings     = np.array(all_embeds, dtype=np.float32)
    labels_arr     = np.array(all_labels, dtype=np.int32)
    severity_arr   = np.array(
        [CLASS_TO_SEVERITY[l] for l in all_labels], dtype=np.int32
    )

    # Verify L2 normalization (should all be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Embedding norms — mean: {norms.mean():.4f}  "
          f"std: {norms.std():.4f}  (should be ~1.0 ± ~0.0001)")

    # Save
    output_path = Path(args.output_dir) / "z_img_all.npz"
    np.savez(
        str(output_path),
        embeddings  = embeddings,
        labels      = labels_arr,
        severity    = severity_arr,
        paths       = np.array(all_paths),
        class_names = np.array(CLASS_LABELS)
    )

    print(f"\n{'=' * 58}")
    print(f"  Embedding Extraction Complete")
    print(f"  {'─' * 48}")
    print(f"  Files processed  : {total}")
    print(f"  Embedding shape  : {embeddings.shape}  (N × 256)")
    print(f"  Saved to         : {output_path}")
    print(f"\n  Contents of .npz:")
    print(f"    embeddings  → ({embeddings.shape[0]}, 256) float32  — z_img vectors")
    print(f"    labels      → ({labels_arr.shape[0]},)     int32    — class index")
    print(f"    severity    → ({severity_arr.shape[0]},)   int32    — severity (0/1/2)")
    print(f"    paths       → ({len(all_paths)},)          str      — source file paths")
    print(f"    class_names → {CLASS_LABELS}")
    print(f"\n  ✅  Hand this .npz to the fusion engine (Aarabhi).")
    print(f"{'=' * 58}\n")


if __name__ == "__main__":
    main()