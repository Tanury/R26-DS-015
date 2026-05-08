from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.seed import stable_hash_int


class EpochDataset(Dataset):
    """Pre-loads all epoch data into memory at construction time.

    This avoids the critical I/O bottleneck of opening and reading the full
    .npz file on every __getitem__ call (which was 50x more disk I/O than needed).
    All epochs are loaded once at __init__ and stored as a flat list of
    (epoch_array, label, subject_id) tuples.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        epochs_dir: str | Path,
        min_epochs: int = 1,
        max_epochs_per_subject: int | None = None,
        seed: int = 42,
    ):
        self.epochs_dir = Path(epochs_dir)
        self.samples: list[tuple[np.ndarray, int, str]] = []
        self.skipped_subjects: list[str] = []
        self.max_epochs_per_subject = max_epochs_per_subject
        self.seed = int(seed)

        for _, row in metadata.iterrows():
            subject_id = str(row["subject_id"])
            path = self.epochs_dir / f"{subject_id}_epochs.npz"
            if not path.exists():
                self.skipped_subjects.append(subject_id)
                continue
            with np.load(path, allow_pickle=True) as data:
                epochs = data["epochs"].astype("float32")  # load ALL epochs once
            if len(epochs) < int(min_epochs):
                self.skipped_subjects.append(subject_id)
                continue
            if max_epochs_per_subject is not None and len(epochs) > int(max_epochs_per_subject):
                rng = np.random.default_rng(self.seed + stable_hash_int(subject_id) % (2**16))
                selected_idx = np.sort(
                    rng.choice(len(epochs), size=int(max_epochs_per_subject), replace=False)
                )
                epochs = epochs[selected_idx]
            label = int(row["label"])
            for i in range(len(epochs)):
                self.samples.append((epochs[i], label, subject_id))

        if not self.samples:
            raise FileNotFoundError(
                f"No preprocessed epoch files found in {self.epochs_dir}. "
                "Run preprocessing first and ensure subjects meet the minimum clean epoch requirement."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        epoch, label, subject_id = self.samples[index]
        x = torch.from_numpy(epoch).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, subject_id


def load_split(split_file: str | Path, fold: int, split: str) -> pd.DataFrame:
    df = pd.read_csv(split_file)
    return df[(df["fold"] == fold) & (df["split"] == split)].reset_index(drop=True)
