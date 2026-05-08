from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay


def save_confusion_matrix(matrix, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["HC", "AD EEG Pattern"], yticklabels=["HC", "AD EEG Pattern"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return str(out)


def save_roc_curve(y_true, y_prob, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return str(out)


def save_probability_bar(hc_probability: float, ad_probability: float, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["Healthy Control", "AD EEG Pattern"], [hc_probability, ad_probability], color=["#3b82f6", "#ef4444"])
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return str(out)


def save_epoch_distribution(epoch_probs, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.hist(epoch_probs, bins=12, color="#6366f1", alpha=0.85)
    plt.xlabel("AD EEG Pattern probability")
    plt.ylabel("Epoch count")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return str(out)


def save_embedding_scatter(
    embeddings,
    labels,
    out_path: str | Path,
    method: str = "pca",
    subject_ids: list[str] | None = None,
) -> str | None:
    if len(embeddings) < 2:
        return None
    method = method.lower()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, len(embeddings) // 3))
        coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
        title = "z_eeg t-SNE projection"
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        title = "z_eeg PCA projection"

    labels_arr = np.asarray(labels)
    colors = {0: "#3b82f6", 1: "#ef4444"}
    names = {0: "Healthy Control", 1: "AD EEG Pattern"}
    plt.figure(figsize=(6, 5))
    for cls in sorted(set(labels)):
        mask = labels_arr == cls
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors.get(int(cls), "#888888"),
            label=names.get(int(cls), str(cls)),
            edgecolor="white",
            linewidth=0.6,
            s=55,
        )
    if subject_ids is not None:
        for i, subject_id in enumerate(subject_ids):
            plt.annotate(subject_id, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.55)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return str(out)


def save_embedding_pca_scatter(embeddings, labels, out_path: str | Path) -> str | None:
    return save_embedding_scatter(embeddings, labels, out_path, method="pca")
