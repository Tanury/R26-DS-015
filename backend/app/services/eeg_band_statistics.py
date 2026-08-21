"""Cohort band-power reference, and the rule for when a "signature" is real.

Band powers come from `eeg_preprocessing`, not from the encoder: the network is fed
raw time-domain epochs `[1, 128, 1024]` and never sees a spectrum. Everything here is
therefore **descriptive** — it says what a diagnostic group's recordings look like,
not what drove a score. The occlusion map in `EegRiskReport.explainability` is the
only causal attribution in this codebase.

The gate exists because on this cohort one of the three conditions has no band-power
signature at all. Measured AUC against healthy controls:

    band                AD     PD     MS
    theta             0.73   0.81   0.53
    alpha             0.39   0.24   0.62
    delta             0.59   0.77   0.29
    theta/alpha       0.72   0.84   0.48

MS sits at chance on the canonical slowing axis while the encoder separates it at
0.992 AUC, and its one separating band (delta) runs *backwards* — MS participants
here are ~33 years younger than every other group, and younger brains show less
slow-wave activity. A panel that always finds something to point at would turn that
age gap into a fabricated neurological finding, so `has_signature` has to be able to
come back False.
"""

from __future__ import annotations

from statistics import median
from typing import Iterable, Sequence


# How a band is expected to move under neurodegenerative slowing. `low_gamma` has no
# agreed direction in this literature, so it can corroborate but never lead.
SLOWING_DIRECTION: dict[str, str] = {
    "delta": "higher",
    "theta": "higher",
    "alpha": "lower",
    "beta": "lower",
    "low_gamma": "none",
    "theta_alpha_ratio": "higher",
}

# The theta/alpha ratio is the single most-reported summary of EEG slowing. Requiring
# it keeps a signature from resting on one incidental band.
CANONICAL_AXIS = "theta_alpha_ratio"

# |AUC - 0.5| below this is not a separation worth showing a reader.
SEPARATION_MARGIN = 0.15

# One band moving is noise; a signature needs corroboration.
MIN_SEPARATING_BANDS = 2

METHOD = (
    "Relative band power per subject, compared across diagnostic groups with a "
    "rank-based AUC. Descriptive only — the encoder consumes raw time-domain epochs "
    "and never sees these values."
)


def auc_vs_reference(values: Sequence[float], reference: Sequence[float]) -> float:
    """Probability that a random `values` sample exceeds a random `reference` sample.

    This is the Mann-Whitney U statistic normalised to [0, 1]. 0.5 means the two
    groups are indistinguishable on this band. Ties score half, so a band that is
    constant in both groups lands on exactly 0.5 rather than 0.0 or 1.0.

    Computed pairwise rather than by ranking: the groups here are 22-35 subjects, so
    the quadratic cost is irrelevant and the definition stays legible.
    """
    if not values or not reference:
        return 0.5
    wins = 0.0
    for value in values:
        for other in reference:
            if value > other:
                wins += 1.0
            elif value == other:
                wins += 0.5
    return wins / (len(values) * len(reference))


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank quantile. Small groups make interpolation false precision."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def _collect(profiles: Iterable[dict[str, float]], band: str) -> list[float]:
    return [float(p[band]) for p in profiles if band in p]


def summarise_group(profiles: Sequence[dict[str, float]], bands: Sequence[str]) -> dict:
    """Median and interquartile range per band for one diagnostic group."""
    return {
        "n": len(profiles),
        "medians": {b: round(median(_collect(profiles, b)), 5) if profiles else 0.0
                    for b in bands},
        "q1": {b: round(_quantile(_collect(profiles, b), 0.25), 5) for b in bands},
        "q3": {b: round(_quantile(_collect(profiles, b), 0.75), 5) for b in bands},
    }


def observed_direction(auc: float, margin: float = SEPARATION_MARGIN) -> str:
    """Which way this group sits relative to controls: higher, lower, or neither."""
    if auc >= 0.5 + margin:
        return "higher"
    if auc <= 0.5 - margin:
        return "lower"
    return "none"


def build_condition_profile(
    condition: str,
    condition_profiles: Sequence[dict[str, float]],
    healthy_profiles: Sequence[dict[str, float]],
    bands: Sequence[str],
    margin: float = SEPARATION_MARGIN,
) -> dict:
    """Per-band separation for one condition, plus the verdict on whether it holds."""
    auc = {
        band: round(auc_vs_reference(_collect(condition_profiles, band),
                                     _collect(healthy_profiles, band)), 4)
        for band in bands
    }
    directions = {band: observed_direction(value, margin) for band, value in auc.items()}

    separating = [b for b in bands if directions[b] != "none"]
    # A separating band that moves against the slowing direction is not evidence for
    # the condition — on this cohort it is the age gap showing through.
    opposing = [
        b for b in separating
        if SLOWING_DIRECTION.get(b, "none") != "none"
        and directions[b] != SLOWING_DIRECTION[b]
    ]
    has_signature = (
        CANONICAL_AXIS in separating and len(separating) >= MIN_SEPARATING_BANDS
    )

    return {
        "condition": condition,
        "n": len(condition_profiles),
        "medians": summarise_group(condition_profiles, bands)["medians"],
        "auc_vs_hc": auc,
        "direction_vs_hc": directions,
        "separating_bands": separating,
        "opposing_bands": opposing,
        "has_signature": has_signature,
        "note": _describe(condition, auc, directions, separating, opposing, has_signature),
    }


def _describe(
    condition: str,
    auc: dict[str, float],
    directions: dict[str, str],
    separating: Sequence[str],
    opposing: Sequence[str],
    has_signature: bool,
) -> str:
    """Plain-language verdict. Written here so the API and UI cannot disagree."""
    pretty = {b: "theta/alpha ratio" if b == CANONICAL_AXIS else b.replace("_", " ")
              for b in auc}

    if has_signature:
        ranked = sorted(separating, key=lambda b: abs(auc[b] - 0.5), reverse=True)
        led = ", ".join(f"{pretty[b]} {directions[b]} (AUC {auc[b]:.2f})" for b in ranked[:3])
        text = (
            f"{len(separating)} of {len(auc)} bands separate {condition} from healthy "
            f"controls in this cohort: {led}. The theta/alpha ratio moves in the "
            f"expected slowing direction, so a band-power comparison is meaningful "
            f"for this condition."
        )
        if opposing:
            against = ", ".join(pretty[b] for b in opposing)
            text += f" Note that {against} runs against the slowing direction."
        return text

    if not separating:
        return (
            f"No frequency band separates {condition} from healthy controls in this "
            f"cohort. Whatever the encoder is using to discriminate {condition}, it is "
            f"not conventional band power, so no pattern is claimed here."
        )

    listed = ", ".join(
        f"{pretty[b]} {directions[b]} (AUC {auc[b]:.2f})" for b in separating
    )
    text = (
        f"{condition} does not show a band-power slowing signature in this cohort: the "
        f"theta/alpha ratio sits at chance (AUC {auc.get(CANONICAL_AXIS, 0.5):.2f}). "
        f"Only {listed} separates at all"
    )
    if opposing:
        against = ", ".join(pretty[b] for b in opposing)
        text += (
            f", and {against} moves opposite to the slowing direction — the profile of a "
            f"younger group rather than an affected one. No pattern is claimed here."
        )
    else:
        text += ", which is not enough to claim a pattern."
    return text


def build_reference(
    profiles_by_class: dict[str, Sequence[dict[str, float]]],
    bands: Sequence[str],
    generated_at: str,
    healthy_class: str = "HC",
    margin: float = SEPARATION_MARGIN,
) -> dict:
    """Assemble the cohort-wide band reference served at `/eeg/band-reference`."""
    healthy = list(profiles_by_class.get(healthy_class, []))
    conditions = {
        name: build_condition_profile(name, list(profiles), healthy, bands, margin)
        for name, profiles in profiles_by_class.items()
        if name != healthy_class
    }
    return {
        "generated_at": generated_at,
        "bands": list(bands),
        "separation_margin": margin,
        "canonical_axis": CANONICAL_AXIS,
        "slowing_direction": {b: SLOWING_DIRECTION.get(b, "none") for b in bands},
        "healthy": summarise_group(healthy, bands),
        "conditions": conditions,
        "method": METHOD,
    }
