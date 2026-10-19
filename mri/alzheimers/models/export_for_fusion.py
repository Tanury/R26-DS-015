"""
export_for_fusion.py
====================
AD MRI branch -> decision-level fusion handoff (Aarabhi A.Y.)

Runs best_model.pth on the 142 held-out test scans listed in
checkpoints/test_split.npz and writes, to outputs/fusion_export/ad_mri/:

    predictions.csv  subject_id, image_id, label, label_name, P_AD, P_MCI, P_CN, pred
    z_img.npz        embeddings (N,256) + subject_ids, image_ids, labels, probs, class_names
    metrics.json     accuracy, balanced accuracy, macro ROC-AUC (one-vs-rest),
                     per-class sensitivity, confusion matrix

It also checks which of the old files (z_img_all.npz / z_img_all_v2.npz)
came from this checkpoint.

The paths in test_split.npz point to Colab, so scans are matched by
filename anywhere under --data_dir.

Run from final_neuro/ with your venv active:
    python mri/alzheimers/models/export_for_fusion.py \
        --data_dir ~/research/vision_encoder/mri/alzheimers/data
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mri.alzheimers.models.vision_encoder import build_encoder  # noqa: E402

CLASS_LABELS = ["AD", "MCI", "CN"]          # must match training order
IMAGE_SIZE = (96, 96, 96)
CKPT_DIR = Path("mri/alzheimers/models/checkpoints")
OLD_EMB = [Path("outputs/alzheimers_mri/z_img_all.npz"),
           Path("outputs/alzheimers_mri/z_img_all_v2.npz")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Folder that contains the preprocessed ADNI scans (searched recursively)")
    p.add_argument("--out_dir", default="outputs/fusion_export/ad_mri")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--use_brain", action="store_true",
                   help="Use *_reg_brain_n4 images instead of the *_reg_brain_mask_n4 files listed in the split")
    p.add_argument("--cpu", action="store_true", help="Force CPU (rule out MPS differences)")
    p.add_argument("--allow_missing", action="store_true",
                   help="Drop test scans that can't be found instead of stopping")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_ids(filename):
    """ADNI_135_S_6840_I1263416_reg_brain_mask_n4.nii.gz -> ('135_S_6840', 'I1263416')"""
    m = re.match(r"ADNI_(\d+_S_\d+)_(I\d+)", filename)
    return (m.group(1), m.group(2)) if m else (filename, "")


def find_local_files(data_dir, wanted_names):
    index = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f in wanted_names and f not in index:
                index[f] = str(Path(root) / f)
    return index


def macro_auc_ovr(y, probs):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except Exception as e:  # sklearn missing or a class absent
        print(f"  (ROC-AUC skipped: {e})")
        return None


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = np.load(CKPT_DIR / "test_split.npz", allow_pickle=True)
    colab_paths = [str(p) for p in split["paths"]]
    labels = split["labels"].astype(int)
    names = [Path(p).name for p in colab_paths]
    if args.use_brain:
        names = [n.replace("_reg_brain_mask_n4", "_reg_brain_n4") for n in names]
        print("Using *_reg_brain_n4 images")

    local = find_local_files(os.path.expanduser(args.data_dir), set(names))
    missing = [n for n in names if n not in local]
    print(f"Test scans: {len(names)} | found locally: {len(local)} | missing: {len(missing)}")
    keep = np.array([n in local for n in names])
    if missing:
        print("Missing (first 10):", *missing[:10], sep="\n  ")
        if not args.allow_missing:
            print("\nRe-run with --allow_missing to drop these, or point --data_dir "
                  "at the folder that holds them.")
            sys.exit(1)
        print(f"\n--allow_missing: dropping {len(missing)} scan(s)")
        names = [n for n, k in zip(names, keep) if k]
        labels = labels[keep]

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")
    model = build_encoder(num_classes=len(CLASS_LABELS), device=device)
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    model.eval()

    tf = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(IMAGE_SIZE)])
    ds = ImageDataset(image_files=[local[n] for n in names], labels=labels.tolist(), transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    embs, probs = [], []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            z, logits = model(x.to(device))
            embs.append(z.cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            print(f"  {min((i + 1) * args.batch_size, len(names))}/{len(names)}", end="\r")
    embs = np.concatenate(embs).astype(np.float32)
    probs = np.concatenate(probs).astype(np.float32)
    preds = probs.argmax(1)
    print()

    subj_ids, img_ids = zip(*[parse_ids(n) for n in names])

    # predictions.csv
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "image_id", "label", "label_name",
                    "P_AD", "P_MCI", "P_CN", "pred", "pred_name"])
        for s, im, y, p, pr in zip(subj_ids, img_ids, labels, probs, preds):
            w.writerow([s, im, int(y), CLASS_LABELS[y], *[f"{v:.6f}" for v in p],
                        int(pr), CLASS_LABELS[pr]])

    # z_img.npz
    np.savez(out / "z_img.npz", embeddings=embs, probs=probs, labels=labels,
             subject_ids=np.array(subj_ids), image_ids=np.array(img_ids),
             class_names=np.array(CLASS_LABELS))

    # metrics.json
    cm = np.zeros((3, 3), dtype=int)
    for y, p in zip(labels, preds):
        cm[y, p] += 1
    recall = {CLASS_LABELS[c]: float(cm[c, c] / cm[c].sum()) for c in range(3)}
    metrics = {
        "branch": "ad_mri",
        "model": "3D ResNet-18 (MONAI) -> 128 -> 256 L2-normalised z_img, linear head",
        "checkpoint": "mri/alzheimers/models/checkpoints/best_model.pth",
        "evaluation": "held-out test split (test_split.npz), subject-level",
        "n": int(len(labels)),
        "dropped_missing_scans": missing,
        "class_order": CLASS_LABELS,
        "class_counts": {CLASS_LABELS[c]: int((labels == c).sum()) for c in range(3)},
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "macro_roc_auc_ovr": macro_auc_ovr(labels, probs),
        "recall_per_class": recall,
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "probabilities": "raw softmax, NOT calibrated",
        "embedding_norm_mean": float(np.linalg.norm(embs, axis=1).mean()),
        "input_files": "_reg_brain_n4" if args.use_brain else "_reg_brain_mask_n4",
        "device": str(device),
        "exported": str(date.today()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Which old file matches this checkpoint?
    print("\nMatch against old embedding files (mean cosine, same subject):")
    for p in OLD_EMB:
        if p.exists():
            old = np.load(p)["embeddings"][keep]
            if old.shape == embs.shape:
                cos = float((old * embs).sum(1).mean())
                tag = "MATCH" if cos > 0.99 else "different model"
                print(f"  {p.name:<20} {cos:.3f}  {tag}")

    print(f"\nAccuracy {metrics['accuracy']:.3f} | balanced {metrics['balanced_accuracy']:.3f} "
          f"| macro AUC {metrics['macro_roc_auc_ovr']}")
    print(f"Recall per class: {recall}")
    print(f"Wrote {out}/predictions.csv, z_img.npz, metrics.json")


if __name__ == "__main__":
    main()
