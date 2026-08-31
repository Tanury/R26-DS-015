"""
test.py  (redesigned)
=====================
Alzheimer's MRI Vision Encoder — Evaluation + Embedding Visualization
----------------------------------------------------------------------
What changed from the original test.py:
    - Loads VisionEncoder instead of raw DenseNet121
    - Extracts z_img embeddings alongside class predictions
    - Adds t-SNE and PCA visualization of the 256-d embedding space
      → This is the key evaluation artifact for the novelty claim:
        showing that z_img clusters by neurodegeneration severity
    - Saves all embeddings to .npz for fusion engine

Run from project root (R26-DS-015/):
    python mri/alzheimers/models/test.py

    Inference on custom folder:
    python mri/alzheimers/models/test.py --infer_dir path/to/niftis/

Author: R26-DS-015 Vision Encoder
"""

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import monai
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

DEFAULT_CHECKPOINT_DIR = "mri/alzheimers/models/checkpoints"
DEFAULT_DATA_DIR       = "mri/alzheimers/data/denoised"
DEFAULT_OUTPUT_DIR     = "outputs/alzheimers_mri"
DEFAULT_IMAGE_SIZE     = (96, 96, 96)
DEFAULT_BATCH_SIZE     = 2

CLASS_LABELS    = ["AD", "MCI", "CN"]
SEVERITY_LABELS = ["CN (0)", "MCI (1)", "AD (2)"]

# Colors for embedding visualization — aligned with project theme
CLASS_COLORS    = {"AD": "#852E47", "MCI": "#FFC64F", "CN": "#519CAB"}
SEVERITY_COLORS = {0: "#519CAB", 1: "#FFC64F", 2: "#852E47"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Alzheimer's MRI VisionEncoder."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data_dir",       type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir",     type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size",     type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--infer_dir",      type=str, default=None,
                        help="Run inference on custom NIfTI folder (no labels needed)")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms():
    return Compose([ScaleIntensity(), EnsureChannelFirst(), Resize(DEFAULT_IMAGE_SIZE)])


def load_model(checkpoint_dir, device):
    ckpt = Path(checkpoint_dir) / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\nRun train.py first."
        )
    model = build_encoder(num_classes=len(CLASS_LABELS), device=device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()
    print(f"  ✅  Loaded: {ckpt}")
    return model


def load_test_split(checkpoint_dir, data_dir):
    split = Path(checkpoint_dir) / "test_split.npz"
    if split.exists():
        data   = np.load(str(split), allow_pickle=True)
        paths  = list(data["paths"])
        labels = list(data["labels"].astype(int))
        print(f"  ✅  Test split: {len(paths)} files")
        return paths, labels

    print(f"  ⚠️   No test_split.npz — loading all files from {data_dir}")
    paths, labels = [], []
    for i, cls in enumerate(CLASS_LABELS):
        d = Path(data_dir) / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.name.endswith(".nii.gz") or f.name.endswith(".nii"):
                paths.append(str(f))
                labels.append(i)
    return paths, labels


def run_evaluation(model, paths, labels, transforms, batch_size, device):
    """Extract predictions, embeddings, and probabilities."""
    ds     = ImageDataset(image_files=paths, labels=labels, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    all_preds   = []
    all_labels  = []
    all_probs   = []
    all_embeds  = []
    softmax     = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in loader:
            inputs  = batch[0].to(device)
            targets = batch[1].to(device)
            z_img, logits = model(inputs)
            probs   = softmax(logits)
            preds   = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeds.extend(z_img.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_embeds)
    )


def print_metrics(preds, labels, probs, output_dir):
    acc = accuracy_score(labels, preds)

    print(f"\n{'=' * 58}")
    print(f"  Evaluation Results")
    print(f"  {'─' * 48}")
    print(f"  Overall Accuracy : {acc:.4f}  ({acc*100:.1f}%)")

    if len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            print(f"  Macro AUC-ROC    : {auc:.4f}")
        except Exception:
            pass

    print(f"\n  Per-class Report:")
    print(classification_report(labels, preds, target_names=CLASS_LABELS, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix (rows=true, cols=predicted):")
    print("         " + "  ".join(f"{l:>5}" for l in CLASS_LABELS))
    for i, row in enumerate(cm):
        print(f"  {CLASS_LABELS[i]:>4}   " + "  ".join(f"{v:>5}" for v in row))
    print(f"{'=' * 58}\n")

    # Save heatmap
    _save_confusion_matrix(cm, output_dir)

    # Save text report
    rpt_path = Path(output_dir) / "test_metrics.txt"
    with open(str(rpt_path), "w") as f:
        f.write("Alzheimer's MRI Vision Encoder — Test Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(labels, preds, target_names=CLASS_LABELS, zero_division=0))
    print(f"  📄  Report saved: {rpt_path}")

    return acc


def _save_confusion_matrix(cm, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Confusion Matrix — Alzheimer's MRI Vision Encoder", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrix: {path}")


def visualize_embeddings(embeddings, labels, output_dir):
    """
    Generate t-SNE and PCA plots of the 256-d z_img embedding space.

    This is the key novelty visualization:
    - Colored by CLASS label → shows disease classification clustering
    - Colored by SEVERITY label → shows neurodegeneration severity clustering

    If the SupCon loss is working correctly, the severity plot should show
    cleaner, more structured clusters than the class plot — because SupCon
    explicitly optimizes for severity-based clustering.
    """
    print(f"\n  Generating embedding visualizations...")

    # Severity labels (derived from class labels)
    class_to_severity = {0: 2, 1: 1, 2: 0}  # AD=2, MCI=1, CN=0
    severity_labels = np.array([class_to_severity[l] for l in labels])

    # ── t-SNE ──────────────────────────────────────────────────────────
    print(f"  Running t-SNE on {embeddings.shape[0]} embeddings (256-d)...")
    tsne   = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                  random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(embeddings)

    # ── PCA ────────────────────────────────────────────────────────────
    pca    = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    # ── Plot: 2 rows × 2 cols ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "z_img Embedding Space — Alzheimer's MRI Vision Encoder\n"
        "(256-d → 2-d projection)",
        fontsize=13, y=1.01
    )

    plot_configs = [
        (axes[0, 0], tsne_2d,  labels,          CLASS_LABELS,
         "t-SNE — Colored by Disease Class", CLASS_COLORS, False),
        (axes[0, 1], tsne_2d,  severity_labels, ["CN (sev=0)", "MCI (sev=1)", "AD (sev=2)"],
         "t-SNE — Colored by Severity (SupCon target)", SEVERITY_COLORS, True),
        (axes[1, 0], pca_2d,   labels,          CLASS_LABELS,
         f"PCA — Colored by Disease Class\n({var_explained:.1f}% variance)", CLASS_COLORS, False),
        (axes[1, 1], pca_2d,   severity_labels, ["CN (sev=0)", "MCI (sev=1)", "AD (sev=2)"],
         f"PCA — Colored by Severity\n({var_explained:.1f}% variance)", SEVERITY_COLORS, True),
    ]

    for ax, coords, plot_labels, label_names, title, colors, is_severity in plot_configs:
        unique_labels = np.unique(plot_labels)
        for ul in unique_labels:
            mask = plot_labels == ul
            if is_severity:
                color = colors[ul]
                name  = label_names[ul]
            else:
                color = list(colors.values())[ul]
                name  = label_names[ul]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, label=name, alpha=0.7, s=40, edgecolors="white", linewidths=0.3
            )
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = Path(output_dir) / "embedding_visualization.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Embedding visualization: {path}")
    print(f"       → Check severity plots: SupCon-trained embeddings should")
    print(f"         cluster by severity (CN/MCI/AD) regardless of disease.")


def save_embeddings(embeddings, labels, paths, output_dir):
    """Save z_img embeddings for the fusion engine."""
    path = Path(output_dir) / "z_img_embeddings.npz"
    np.savez(
        str(path),
        embeddings=embeddings,
        labels=np.array(labels),
        paths=np.array(paths)
    )
    print(f"  💾  z_img embeddings saved: {path}")
    print(f"       Shape: {embeddings.shape}  — ready for fusion engine")


def run_inference_mode(model, infer_dir, transforms, batch_size, device, output_dir):
    """Inference on unlabelled files."""
    files = []
    for root, _, fnames in os.walk(infer_dir):
        for f in sorted(fnames):
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                files.append(str(Path(root) / f))

    if not files:
        print(f"❌  No NIfTI files in {infer_dir}")
        return

    print(f"  Found {len(files)} files for inference.")
    dummy = [0] * len(files)
    ds    = ImageDataset(image_files=files, labels=dummy, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    all_preds  = []
    all_embeds = []
    softmax    = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            z_img, logits = model(inputs)
            probs  = softmax(logits)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_embeds.extend(z_img.cpu().numpy())

    print(f"\n{'=' * 58}")
    print(f"  Inference Results")
    print(f"  {'─' * 48}")
    print(f"  {'File':<42} {'Pred':>6}  {'Conf':>8}")
    print(f"  {'─' * 48}")
    for fpath, pred, emb in zip(files, all_preds, all_embeds):
        fname = Path(fpath).name[:41]
        label = CLASS_LABELS[pred]
        conf  = softmax(torch.tensor(emb).unsqueeze(0)).squeeze()[pred].item() * 100
        print(f"  {fname:<42} {label:>6}  {conf:>7.1f}%")
    print(f"{'=' * 58}\n")

    save_embeddings(np.array(all_embeds), all_preds, files, output_dir)


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 58)
    print("  Alzheimer's MRI Vision Encoder — Evaluation")
    print("=" * 58)

    device     = get_device()
    model      = load_model(args.checkpoint_dir, device)
    transforms = get_transforms()

    if args.infer_dir:
        print(f"\n  Inference mode: {args.infer_dir}")
        run_inference_mode(
            model, args.infer_dir, transforms,
            args.batch_size, device, args.output_dir
        )
        return

    print(f"\n  Loading test data...")
    paths, labels = load_test_split(args.checkpoint_dir, args.data_dir)
    if not paths:
        print("❌  No test files found. Run train.py first.")
        return

    print(f"\n  Running evaluation ({len(paths)} files)...")
    preds, labels_arr, probs, embeds = run_evaluation(
        model, paths, labels, transforms, args.batch_size, device
    )

    print_metrics(preds, labels_arr, probs, args.output_dir)
    visualize_embeddings(embeds, labels_arr, args.output_dir)
    save_embeddings(embeds, labels_arr, paths, args.output_dir)

    print(f"\n  ✅  All outputs saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()