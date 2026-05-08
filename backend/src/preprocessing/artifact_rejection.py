from __future__ import annotations

import numpy as np


def apply_fastica_if_enabled(raw, cfg: dict, subject_id: str | None = None):
    """Optionally apply a conservative FastICA cleaning pass.

    Phase 1 keeps ICA disabled by default because amplitude rejection is more
    stable for a small baseline. When enabled, this uses MNE's FastICA and only
    excludes explicitly configured components or components strongly correlated
    with frontal EEG channels.
    """
    enabled = bool(cfg.get("use_ica", False))
    summary = {
        "enabled": enabled,
        "applied": False,
        "method": "FastICA",
        "excluded_components": [],
        "reason": "Disabled in config.",
    }
    if not enabled:
        return raw, summary

    try:
        from mne.preprocessing import ICA
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        summary["reason"] = f"MNE ICA import failed: {type(exc).__name__}: {exc}"
        return raw, summary

    n_components = cfg.get("ica_n_components", "auto")
    if n_components in (None, "auto"):
        n_components = min(len(raw.ch_names), 20)
    n_components = max(1, min(int(n_components), len(raw.ch_names)))

    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=int(cfg.get("ica_random_state", 42)),
        max_iter=int(cfg.get("ica_max_iter", 200)),
    )
    fit_raw = raw.copy()
    ica.fit(fit_raw, picks="eeg", verbose=False)

    configured = [int(i) for i in cfg.get("ica_exclude_components", [])]
    automatic = _frontal_correlated_components(
        fit_raw,
        ica,
        threshold=float(cfg.get("ica_frontal_corr_threshold", 0.35)),
        max_components=int(cfg.get("ica_max_auto_exclude", 2)),
    )
    excluded = sorted(set(configured + automatic))
    summary["excluded_components"] = excluded

    if not excluded:
        summary["reason"] = (
            "FastICA fitted, but no components met the configured exclusion rule; "
            "signal left unchanged."
        )
        return raw, summary

    cleaned = raw.copy()
    ica.exclude = excluded
    ica.apply(cleaned, verbose=False)
    summary["applied"] = True
    summary["reason"] = "FastICA components excluded and signal reconstructed."
    return cleaned, summary


def _frontal_correlated_components(raw, ica, threshold: float, max_components: int) -> list[int]:
    frontal_names = {"fp1", "fp2", "f7", "f8"}
    frontal_picks = [
        i for i, name in enumerate(raw.ch_names)
        if name.strip().lower().replace(".", "") in frontal_names
    ]
    if not frontal_picks:
        return []

    frontal_signal = raw.get_data(picks=frontal_picks).mean(axis=0)
    sources = ica.get_sources(raw).get_data()
    scored = []
    for idx, component in enumerate(sources):
        corr = np.corrcoef(component, frontal_signal)[0, 1]
        if np.isfinite(corr) and abs(float(corr)) >= threshold:
            scored.append((abs(float(corr)), idx))
    scored.sort(reverse=True)
    return [idx for _score, idx in scored[:max_components]]


def reject_by_amplitude(epochs: np.ndarray, threshold_uv: float) -> tuple[np.ndarray, np.ndarray]:
    if epochs.shape[0] == 0:
        return epochs, np.zeros(0, dtype=bool)
    threshold_v = threshold_uv * 1e-6
    peak_to_peak = np.ptp(epochs, axis=2).max(axis=1)
    clean_mask = peak_to_peak <= threshold_v
    return epochs[clean_mask], clean_mask


def signal_quality(clean_ratio: float) -> str:
    if clean_ratio >= 0.8:
        return "Good"
    if clean_ratio >= 0.5:
        return "Moderate"
    return "Poor"
