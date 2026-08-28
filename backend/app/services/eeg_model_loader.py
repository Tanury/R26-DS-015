"""Load the exported TorchScript encoder and its joblib side-bundle.

TorchScript is used deliberately: `torch.jit.load()` reconstructs the computation
graph without needing `NeuroRiskEncoder` or any other notebook class to be
importable. The backend therefore has nothing to keep in sync with training code.

PyTorch is imported lazily. Tier 1 (cohort browsing) must keep working on a
deployment that never installs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.exceptions import EegBundleError


TORCHSCRIPT_FILENAME = "neuro_risk_encoder.torchscript.pt"
BUNDLE_FILENAME = "neuro_risk_inference_bundle.joblib"


@dataclass(frozen=True)
class EegModelAssets:
    model: Any
    risk_conditions: tuple[str, ...]
    class_names: tuple[str, ...]
    input_shape: tuple[int, int, int]
    channel_order: tuple[str, ...]
    sampling_rate_hz: float
    epoch_length_seconds: float
    bands: dict[str, list[float]]
    risk_bands: dict[str, float]
    class_centroids: dict[str, np.ndarray]
    preprocessing_config: dict[str, Any]
    representation_config: dict[str, Any]
    representation: str
    # Which z-score variant the model was fitted under. Bundles exported before
    # the epsilon fix do not declare it, and must be served as "legacy_eps".
    standardization: str


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise EegBundleError(
            "PyTorch is required for EEG inference but is not installed. "
            "Install the CPU wheel: pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from exc


@lru_cache(maxsize=1)
def load_eeg_assets() -> EegModelAssets:
    torch = _torch()
    model_dir = Path(settings.eeg_model_dir)

    try:
        import joblib

        bundle = joblib.load(model_dir / BUNDLE_FILENAME)
    except FileNotFoundError as exc:
        raise EegBundleError(f"Missing EEG bundle file: {BUNDLE_FILENAME}") from exc
    except Exception as exc:
        raise EegBundleError(f"Unable to read the EEG inference bundle: {exc}") from exc

    try:
        model = torch.jit.load(str(model_dir / TORCHSCRIPT_FILENAME), map_location="cpu")
        model.eval()
    except Exception as exc:
        raise EegBundleError(f"Unable to load the TorchScript encoder: {exc}") from exc

    # A burst of EEG jobs must not starve the speech endpoints sharing this process.
    torch.set_num_threads(min(2, torch.get_num_threads()))

    try:
        input_shape = tuple(int(v) for v in bundle["input_shape"])
        assets = EegModelAssets(
            model=model,
            risk_conditions=tuple(str(c) for c in bundle["risk_conditions"]),
            class_names=tuple(str(c) for c in bundle.get("class_names", [])),
            input_shape=input_shape,  # type: ignore[arg-type]
            channel_order=tuple(str(c) for c in bundle["channel_order"]),
            sampling_rate_hz=float(bundle["sampling_rate_hz"]),
            epoch_length_seconds=float(bundle["epoch_length_seconds"]),
            bands={str(k): [float(v[0]), float(v[1])] for k, v in bundle["bands"].items()},
            risk_bands={str(k): float(v) for k, v in bundle["risk_bands"].items()},
            class_centroids={
                str(k): np.asarray(v, dtype="float32")
                for k, v in (bundle.get("class_centroids") or {}).items()
            },
            preprocessing_config=dict(bundle.get("preprocessing_config") or {}),
            representation_config=dict(bundle.get("representation_config") or {}),
            representation=str(bundle.get("representation", "hybrid")),
            standardization=str(bundle.get("standardization", "legacy_eps")),
        )
    except KeyError as exc:
        raise EegBundleError(f"EEG bundle is missing required key: {exc}") from exc

    if len(assets.input_shape) != 3:
        raise EegBundleError(f"EEG bundle input_shape must be 3-D, got {assets.input_shape}")
    if not assets.risk_conditions:
        raise EegBundleError("EEG bundle declares no risk conditions.")
    if assets.standardization not in {"legacy_eps", "zscore_exact"}:
        raise EegBundleError(
            f"EEG bundle declares unknown standardization {assets.standardization!r}"
        )

    return assets


def reset_cache() -> None:
    load_eeg_assets.cache_clear()
