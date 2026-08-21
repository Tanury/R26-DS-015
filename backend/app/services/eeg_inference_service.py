"""Turn preprocessed epochs into an EegRiskReport.

Stage 4 (the hybrid STFT + Morlet CWT tensor) lives here rather than in
preprocessing because it defines the model input contract and must move with the
model, not with the signal cleaning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.exceptions import EegBundleError, PredictionError
from app.schemas.eeg_assessment import (
    ConditionRisk,
    EegRiskReport,
    EmbeddingSummary,
    Explainability,
    FourClassPrediction,
    RiskAssessment,
    SignalQuality,
)
from app.services.eeg_cohort_service import load_model_card
from app.services.eeg_model_loader import EegModelAssets, load_eeg_assets
from app.services.eeg_preprocessing import REGION_OF_BANK


def _pool_to_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    n = x.shape[-1]
    if n == n_bins:
        return x
    edges = np.linspace(0, n, n_bins + 1)
    out = np.empty(x.shape[:-1] + (n_bins,), dtype="float32")
    for b in range(n_bins):
        lo, hi = int(np.floor(edges[b])), max(int(np.ceil(edges[b + 1])), int(np.floor(edges[b])) + 1)
        out[..., b] = x[..., lo:min(hi, n)].mean(axis=-1)
    return out


def build_representation(epochs: np.ndarray, assets: EegModelAssets) -> np.ndarray:
    """[n, C, T] -> [n, F, C, bins], matching the notebook's Stage 4 exactly."""
    representation = assets.representation
    if representation == "raw":
        return epochs[:, None, :, :].astype("float32")

    cfg = assets.representation_config
    sfreq = assets.sampling_rate_hz
    n_bins = int(cfg.get("n_time_bins", 16))
    band_edges = [(name, lo, hi) for name, (lo, hi) in assets.bands.items()]
    parts: list[np.ndarray] = []

    if representation in ("stft", "hybrid"):
        from scipy.signal import stft as scipy_stft

        nperseg = max(8, int(round(float(cfg.get("stft_window_seconds", 1.0)) * sfreq)))
        noverlap = int(round(nperseg * float(cfg.get("stft_overlap", 0.75))))
        freqs, _t, Z = scipy_stft(
            epochs, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
            axis=-1, boundary=None, padded=False,
        )
        power = (np.abs(Z) ** 2).astype("float32")
        planes = []
        for _name, lo, hi in band_edges:
            sel = (freqs >= lo) & (freqs < hi)
            band = (power[:, :, sel, :].mean(axis=2) if sel.any()
                    else np.zeros(power.shape[:2] + power.shape[3:], dtype="float32"))
            planes.append(_pool_to_bins(band, n_bins))
        parts.append(np.stack(planes, axis=1))

    if representation in ("cwt", "hybrid"):
        from mne.time_frequency import tfr_array_morlet

        freqs = np.logspace(
            np.log10(float(cfg.get("cwt_min_freq_hz", 2.0))),
            np.log10(float(cfg.get("cwt_max_freq_hz", 40.0))),
            int(cfg.get("cwt_n_freqs", 24)),
        )
        n_cycles = np.clip(freqs / 2.0, 3.0, 12.0)
        chunk = max(1, int(cfg.get("cwt_chunk_epochs", 8)))
        blocks = []
        for start in range(0, epochs.shape[0], chunk):
            power = tfr_array_morlet(
                epochs[start:start + chunk].astype("float64"), sfreq=sfreq,
                freqs=freqs, n_cycles=n_cycles, output="power", verbose=False,
            )
            planes = []
            for _name, lo, hi in band_edges:
                sel = (freqs >= lo) & (freqs < hi)
                band = (power[:, :, sel, :].mean(axis=2) if sel.any()
                        else np.zeros(power.shape[:2] + power.shape[3:]))
                planes.append(_pool_to_bins(band.astype("float32"), n_bins))
            blocks.append(np.stack(planes, axis=1))
        parts.append(np.concatenate(blocks, axis=0))

    if not parts:
        raise EegBundleError(f"Unknown input representation: {representation!r}")

    tensor = np.concatenate(parts, axis=1)
    if cfg.get("log_power", True):
        tensor = np.log10(tensor + 1e-12)
    mean = tensor.mean(axis=(2, 3), keepdims=True)
    std = tensor.std(axis=(2, 3), keepdims=True)
    return ((tensor - mean) / (std + 1e-6)).astype("float32")


def risk_band(score: float, bands: dict[str, float]) -> str:
    if score <= float(bands.get("low_max", 0.39)):
        return "Low"
    return "Medium" if score <= float(bands.get("medium_max", 0.69)) else "High"


def _band_power_profile(epochs: np.ndarray, sfreq: float, bands: dict[str, list[float]]) -> dict[str, float]:
    from scipy.signal import welch

    freqs, psd = welch(epochs[:40], fs=sfreq, nperseg=min(epochs.shape[-1], int(sfreq * 2)), axis=-1)
    psd_mean = psd.mean(axis=(0, 1))
    in_band = (freqs >= 0.5) & (freqs <= 40.0)
    total = float(psd_mean[in_band].sum())
    profile: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        profile[name] = round(float(psd_mean[sel].sum() / total), 5) if total > 0 else 0.0
    alpha = profile.get("alpha", 0.0)
    profile["theta_alpha_ratio"] = round(profile.get("theta", 0.0) / alpha, 5) if alpha else 0.0
    return profile


def _occlusion(model: Any, tensor: Any, target: int, channel_order: tuple[str, ...],
               band_names: list[str], n_planes: int) -> Explainability:
    """Zero one region / band plane at a time and record the drop in target risk."""
    import torch

    with torch.no_grad():
        base = float(model(tensor)[0].mean(dim=0)[target])

        regions: dict[str, float] = {}
        for bank, region in REGION_OF_BANK.items():
            idx = [i for i, c in enumerate(channel_order) if c.startswith(bank)]
            if not idx:
                continue
            occluded = tensor.clone()
            occluded[:, :, idx, :] = 0.0
            regions[region] = round(base - float(model(occluded)[0].mean(dim=0)[target]), 4)

        bands: dict[str, float] = {}
        if n_planes > 1 and band_names:
            per_plane = []
            for plane in range(n_planes):
                occluded = tensor.clone()
                occluded[:, plane, :, :] = 0.0
                per_plane.append(base - float(model(occluded)[0].mean(dim=0)[target]))
            # Hybrid stacks STFT then CWT over the same band order.
            for i, name in enumerate(band_names):
                values = [per_plane[j] for j in range(i, n_planes, len(band_names))]
                bands[name] = round(float(np.mean(values)), 4)

    return Explainability(scalp_region_importance=regions, band_importance=bands)


def infer(preprocessed: dict[str, Any], subject_id: str, explain: bool = True) -> EegRiskReport:
    """Run the encoder over preprocessed epochs and assemble the report."""
    import torch

    assets = load_eeg_assets()
    epochs = preprocessed["epochs"]
    quality = preprocessed["quality"]

    tensor_np = build_representation(epochs, assets)
    expected = assets.input_shape
    actual = tuple(int(v) for v in tensor_np.shape[1:])
    if actual != tuple(expected):
        raise PredictionError(
            f"Preprocessed tensor {actual} does not match the model's expected "
            f"input {tuple(expected)}. The recording geometry is incompatible."
        )

    tensor = torch.from_numpy(tensor_np)
    with torch.no_grad():
        risk_scores_t, class_probs_t, z_eeg_t = assets.model(tensor)

    epoch_risk = risk_scores_t.cpu().numpy()
    risk = epoch_risk.mean(axis=0)
    class_probs = class_probs_t.cpu().numpy().mean(axis=0)

    z_epochs = z_eeg_t.cpu().numpy()
    z_mean = z_epochs.mean(axis=0)
    raw_norm = float(np.linalg.norm(z_mean))
    z_eeg = z_mean / max(raw_norm, 1e-12)

    card = load_model_card()
    severity = card.confound_disclosure.severity_by_condition

    conditions: dict[str, ConditionRisk] = {}
    for j, name in enumerate(assets.risk_conditions):
        score = float(risk[j])
        conditions[name] = ConditionRisk(
            risk_score=round(score, 4),
            risk_band=risk_band(score, assets.risk_bands),  # type: ignore[arg-type]
            label=f"{name}-related EEG risk pattern",
            epoch_score_std=round(float(epoch_risk[:, j].std()), 4),
            epoch_score_range=[round(float(epoch_risk[:, j].min()), 4),
                               round(float(epoch_risk[:, j].max()), 4)],
            confound_severity=severity.get(name, "unknown"),
        )

    cosines = {
        name: round(float(np.dot(z_eeg, centroid)), 4)
        for name, centroid in assets.class_centroids.items()
    }
    nearest = max(cosines, key=cosines.get) if cosines else None

    explainability = Explainability()
    if explain:
        try:
            explainability = _occlusion(
                assets.model, tensor, int(np.argmax(risk)), assets.channel_order,
                list(assets.bands), int(tensor_np.shape[1]),
            )
        except Exception:  # explainability is a nicety, never a failure mode
            explainability = Explainability(method="occlusion unavailable for this run")

    return EegRiskReport(
        subject_id=subject_id,
        source="upload",
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset={"name": "user upload", "task": "resting-state EEG"},
        risk_scores={
            f"{name.lower()}_risk_score": round(float(risk[j]), 4)
            for j, name in enumerate(assets.risk_conditions)
        },
        risk_assessment=RiskAssessment(
            conditions=conditions,
            highest_risk_condition=assets.risk_conditions[int(np.argmax(risk))],
            interpretation=(
                "Each score is an independent probability that the recording shows the EEG "
                "pattern associated with that condition. Scores do not sum to 1 and are not "
                "mutually exclusive: elevated scores on more than one condition are "
                "meaningful, not contradictory. These are decision-support indicators, not "
                "diagnoses."
            ),
            risk_bands=assets.risk_bands,
        ),
        optional_four_class_prediction=(
            FourClassPrediction(
                predicted_class=assets.class_names[int(np.argmax(class_probs))],
                class_probabilities={
                    name: round(float(class_probs[i]), 4)
                    for i, name in enumerate(assets.class_names)
                },
            ) if assets.class_names else None
        ),
        signal_quality=SignalQuality(**quality),
        band_power_profile=_band_power_profile(epochs, assets.sampling_rate_hz, assets.bands),
        embedding=EmbeddingSummary(
            dim=int(z_eeg.shape[0]),
            l2_norm=round(float(np.linalg.norm(z_eeg)), 6),
            availability_flag=1,
            consistency=round(min(raw_norm, 1.0), 4),
            cosine_to_class_centroids=cosines,
            nearest_centroid=nearest,
        ),
        explainability=explainability,
        confound_disclosure=card.confound_disclosure,
        model_summary={
            "architecture": card.architecture,
            "input_representation": assets.representation,
            "embedding_dim": int(z_eeg.shape[0]),
            "run_id": card.run_id,
        },
        clinical_disclaimer=settings.eeg_disclaimer,
    )
