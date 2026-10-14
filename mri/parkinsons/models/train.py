
import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    Compose, EnsureChannelFirst, Resize, ScaleIntensity,
    RandRotate90, RandFlip, RandZoom, RandGaussianNoise,
    RandAffine, NormalizeIntensity, CropForeground,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.utils.losses import CombinedLoss
from mri.alzheimers.models.vision_encoder import build_encoder  # same architecture


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_DATA_DIR       = "mri/parkinsons/data/denoised"
DEFAULT_CHECKPOINT_DIR = "mri/parkinsons/models/checkpoints"
DEFAULT_LOG_DIR        = "logs/parkinsons_mri"
DEFAULT_EPOCHS         = 50
DEFAULT_BATCH_SIZE     = 4
DEFAULT_LR             = 1e-4
DEFAULT_VAL_INTERVAL   = 2
DEFAULT_IMAGE_SIZE     = (96, 96, 96)
DEFAULT_ALPHA          = 0.5
DEFAULT_SUPCON_TEMP    = 0.07

# Binary classification — PD vs HC
CLASS_LABELS = ["PD", "HC"]
NUM_CLASSES  = 2

# Severity for SupCon: HC (normal) = 0, PD (disease) = 1
CLASS_TO_SEVERITY = {
    0: 1,   # PD  → severity 1
    1: 0,   # HC  → severity 0
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VisionEncoder for Parkinson's MRI classification (PD vs HC)."
    )
    parser.add_argument("--data_dir",        type=str,   default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint_dir",  type=str,   default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log_dir",         type=str,   default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--alpha",           type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--supcon_temp",     type=float, default=DEFAULT_SUPCON_TEMP)
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--pretrained_path", type=str,   default=None,
                        help="Path to MedicalNet resnet_18.pth (optional)")
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple MPS (M-series GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")
    return device


def load_data(data_dir: str) -> tuple:
    images, labels = [], []
    print(f"\n  Class distribution:")
    for label_idx, label_name in enumerate(CLASS_LABELS):
        cls_dir = Path(data_dir) / label_name
        if not cls_dir.exists():
            print(f"    {label_name}: ⚠️  not found at {cls_dir}")
            continue
        files = sorted([
            f for f in cls_dir.iterdir()
            if f.name.endswith(".nii.gz") or f.name.endswith(".nii")
        ])
        for f in files:
            images.append(str(f))
            labels.append(label_idx)
        print(f"    {label_name} (class {label_idx}, "
              f"severity {CLASS_TO_SEVERITY[label_idx]}): {len(files)} files")
    return images, labels


def build_transforms(image_size: tuple) -> tuple:
    train_tf = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        CropForeground(),
        Resize(image_size),
        RandAffine(
            prob=0.6,
            rotate_range=(0.15, 0.15, 0.15),
            translate_range=(8, 8, 8),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="zeros",
        ),
        RandFlip(prob=0.4, spatial_axis=0),
        RandFlip(prob=0.3, spatial_axis=1),
        RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.4),
        RandGaussianNoise(prob=0.25, mean=0.0, std=0.02),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
    val_tf = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        CropForeground(),
        Resize(image_size),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
    return train_tf, val_tf


def build_dataloaders(images, labels, train_tf, val_tf, batch_size) -> tuple:
    train_x, temp_x, train_y, temp_y = train_test_split(
        images, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=42, stratify=temp_y
    )
    print(f"\n  Split → Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}")

    pin_memory  = torch.cuda.is_available()
    num_workers = 0

    # Balanced sampler
    class_counts   = [train_y.count(i) for i in range(NUM_CLASSES)]
    class_weights  = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[l] for l in train_y]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    print(f"  Class counts (train): {dict(zip(CLASS_LABELS, class_counts))}")

    train_loader = DataLoader(
        ImageDataset(image_files=train_x, labels=train_y, transform=train_tf),
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        ImageDataset(image_files=val_x, labels=val_y, transform=val_tf),
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        ImageDataset(image_files=test_x, labels=test_y, transform=val_tf),
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )

    test_info = {"paths": test_x, "labels": test_y}
    return train_loader, val_loader, test_loader, test_info


def get_severity_labels(class_labels: torch.Tensor) -> torch.Tensor:
    severity = torch.zeros_like(class_labels)
    for cls, sev in CLASS_TO_SEVERITY.items():
        severity[class_labels == cls] = sev
    return severity


def train_model(model, train_loader, val_loader, device, args, writer):
    criterion = CombinedLoss(alpha=args.alpha, temperature=args.supcon_temp)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pth"
    encoder_path    = Path(args.checkpoint_dir) / "best_encoder.pth"

    best_val_acc  = -1.0
    best_epoch    = -1
    epoch_losses  = []
    ce_losses     = []
    supcon_losses = []
    val_accs      = []

    print(f"\n  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Alpha (CE)  : {args.alpha}  |  SupCon: {1 - args.alpha}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Classes     : {CLASS_LABELS} (binary)")
    print(f"\n  TensorBoard : tensorboard --logdir {os.path.abspath(args.log_dir)}\n")

    for epoch in range(args.epochs):
        print(f"\n{'─' * 58}")
        print(f"  Epoch {epoch+1}/{args.epochs}  |  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'─' * 58}")

        model.train()
        epoch_total = epoch_ce = epoch_sc = 0.0
        step = 0
        n_batches = len(train_loader)

        for batch in train_loader:
            step   += 1
            inputs  = batch[0].to(device)
            cls_lbl = batch[1].to(device)
            sev_lbl = get_severity_labels(cls_lbl).to(device)

            optimizer.zero_grad()
            z_img, logits = model(inputs)
            total, ce_val, sc_val = criterion(z_img, logits, cls_lbl, sev_lbl)

            if not torch.isfinite(total):
                optimizer.zero_grad()
                continue

            total.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            epoch_total += total.item()
            epoch_ce    += ce_val.item()
            epoch_sc    += sc_val.item() if torch.isfinite(sc_val) else 0.0

            print(f"  [{step:>3}/{n_batches}] "
                  f"loss: {total.item():.4f}  "
                  f"CE: {ce_val.item():.4f}  "
                  f"SupCon: {sc_val.item():.4f}")

            global_step = n_batches * epoch + step
            writer.add_scalar("Loss/train_total",  total.item(),  global_step)
            writer.add_scalar("Loss/train_CE",     ce_val.item(), global_step)
            writer.add_scalar("Loss/train_SupCon",
                              sc_val.item() if torch.isfinite(sc_val) else 0.0,
                              global_step)

        scheduler.step()
        avg_t = epoch_total / max(step, 1)
        avg_c = epoch_ce    / max(step, 1)
        avg_s = epoch_sc    / max(step, 1)
        epoch_losses.append(avg_t)
        ce_losses.append(avg_c)
        supcon_losses.append(avg_s)

        # Validate
        if (epoch + 1) % DEFAULT_VAL_INTERVAL == 0:
            model.eval()
            correct = total_n = 0
            with torch.no_grad():
                for vb in val_loader:
                    vi = vb[0].to(device)
                    vl = vb[1].to(device)
                    _, vlogits = model(vi)
                    correct  += vlogits.argmax(1).eq(vl).sum().item()
                    total_n  += len(vl)

            val_acc = correct / max(total_n, 1)
            val_accs.append(val_acc)
            writer.add_scalar("Accuracy/validation", val_acc, epoch + 1)

            print(f"\n  Val accuracy: {val_acc:.4f}  "
                  f"(best: {best_val_acc:.4f} @ epoch {best_epoch})")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch + 1
                torch.save(model.state_dict(), str(checkpoint_path))
                torch.save(model.mri_encoder.state_dict(), str(encoder_path))
                print(f"  ✅  New best! → {checkpoint_path}")

    return best_val_acc, best_epoch, epoch_losses, ce_losses, supcon_losses, val_accs


def evaluate_test(model, test_loader, device, checkpoint_dir):
    ckpt = Path(checkpoint_dir) / "best_model.pth"
    if not ckpt.exists():
        print("  ⚠️  No checkpoint found.")
        return

    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    all_preds  = []
    all_labels = []
    all_embeds = []
    all_probs  = []

    with torch.no_grad():
        for batch in test_loader:
            inputs  = batch[0].to(device)
            targets = batch[1].to(device)
            z_img, logits = model(inputs)
            probs   = torch.softmax(logits, dim=1)
            preds   = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_embeds.extend(z_img.cpu().numpy())
            all_probs.extend(probs[:, 0].cpu().numpy())  # PD probability

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    embeds = np.array(all_embeds)
    probs  = np.array(all_probs)

    acc = accuracy_score(labels, preds)
    print(f"\n{'=' * 58}")
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # AUC for binary classification
    try:
        auc = roc_auc_score(labels, probs)
        print(f"  AUC-ROC     : {auc:.4f}")
    except Exception:
        pass

    print(f"\n  Per-class report:")
    print(classification_report(
        labels, preds, target_names=CLASS_LABELS, zero_division=0
    ))
    print(f"  z_img shape: {embeds.shape}  (ready for fusion engine)")
    print(f"{'=' * 58}\n")

    # Save embeddings
    embed_path = Path(checkpoint_dir) / "test_embeddings.npz"
    np.savez(str(embed_path), embeddings=embeds, labels=labels, probs=probs)
    print(f"  💾  Test embeddings saved: {embed_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_title("Confusion Matrix — Parkinson's MRI Vision Encoder")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = Path(args_global.log_dir) / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150)
    plt.close()
    print(f"  📊  Confusion matrix saved: {cm_path}")


def save_curves(epoch_losses, ce_losses, supcon_losses,
                val_accs, val_interval, log_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep = range(1, len(epoch_losses) + 1)

    axes[0].set_title("Total Loss", fontsize=12)
    axes[0].plot(ep, epoch_losses, color="#C0392B", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("CE vs SupCon Loss", fontsize=12)
    axes[1].plot(ep, ce_losses,     color="#E74C3C", linewidth=2, label="CrossEntropy")
    axes[1].plot(ep, supcon_losses, color="#C0C0C0", linewidth=2, label="SupCon")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    val_ep = [(i + 1) * val_interval for i in range(len(val_accs))]
    axes[2].set_title("Validation Accuracy", fontsize=12)
    axes[2].plot(val_ep, val_accs, color="#C0392B", linewidth=2, marker="o")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    plt.suptitle(
        "Parkinson's MRI Vision Encoder — Training (PD vs HC)", fontsize=13
    )
    plt.tight_layout()
    path = Path(log_dir) / "training_curves.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Training curves saved: {path}")


# Global reference for evaluate_test
args_global = None


def main():
    global args_global
    args = parse_args()
    args_global = args

    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    print("\n" + "=" * 58)
    print("  Parkinson's MRI — Vision Encoder Training (PD vs HC)")
    print("  Loss: CrossEntropy + Supervised Contrastive (SupCon)")
    print("=" * 58)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Device...")
    device = get_device()

    print(f"\n[2/6] Loading data from: {args.data_dir}")
    images, labels = load_data(args.data_dir)

    if len(images) == 0:
        print("❌  No data found. Run compile.py + preprocessing first.")
        return

    print(f"  Total: {len(images)} files")
    for i, cls in enumerate(CLASS_LABELS):
        count = labels.count(i)
        if count < 5:
            print(f"  ⚠️  Only {count} files for {cls}. Need ≥5 per class.")
            return

    print(f"\n[3/6] Transforms...")
    train_tf, val_tf = build_transforms(DEFAULT_IMAGE_SIZE)

    print(f"\n[4/6] DataLoaders (batch: {args.batch_size})...")
    train_loader, val_loader, test_loader, test_info = build_dataloaders(
        images, labels, train_tf, val_tf, args.batch_size
    )

    np.savez(
        str(Path(args.checkpoint_dir) / "test_split.npz"),
        paths=np.array(test_info["paths"]),
        labels=np.array(test_info["labels"])
    )

    print(f"\n[5/6] Building VisionEncoder (binary: {NUM_CLASSES} classes)...")
    model = build_encoder(
        num_classes=NUM_CLASSES,
        freeze_backbone=args.freeze_backbone,
        pretrained_path=args.pretrained_path,
        device=device,
    )

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"  ✅  Resumed from: {args.resume}")

    writer = SummaryWriter(log_dir=args.log_dir)

    print(f"\n[6/6] Training...")
    best_acc, best_ep, tot_l, ce_l, sc_l, val_a = train_model(
        model, train_loader, val_loader, device, args, writer
    )
    writer.close()

    print(f"\n{'=' * 58}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_acc:.4f} at epoch {best_ep}")
    print(f"{'=' * 58}")

    save_curves(tot_l, ce_l, sc_l, val_a, DEFAULT_VAL_INTERVAL, args.log_dir)

    print(f"\n  Running test evaluation...")
    evaluate_test(model, test_loader, device, args.checkpoint_dir)

    print(f"\n  ✅  Done. Run embed.py to extract z_img for fusion engine.\n")


if __name__ == "__main__":
    main()