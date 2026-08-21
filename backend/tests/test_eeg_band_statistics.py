"""The band-power reference, and the gate that stops it inventing a finding.

The point of these tests is the negative case. A panel that always finds a pattern
is worse than no panel, because on this cohort MS has no band-power signature at all
— its one separating band runs backwards, tracking the 33-year age gap rather than
the disease. `has_signature` has to come back False there, and stay False if someone
later loosens the margin without thinking about it.
"""

import json
from pathlib import Path

import pytest

from app.services.eeg_band_statistics import (
    CANONICAL_AXIS,
    auc_vs_reference,
    build_condition_profile,
    build_reference,
    observed_direction,
)


BANDS = ["delta", "theta", "alpha", "beta", "low_gamma", "theta_alpha_ratio"]
REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app" / "models" / "eeg" / "cohort" / "band_reference.json"
)


def _profiles(values: dict[str, list[float]]) -> list[dict[str, float]]:
    """Transpose {band: [per-subject]} into the per-subject dicts the builder takes."""
    n = len(next(iter(values.values())))
    return [{band: series[i] for band, series in values.items()} for i in range(n)]


def test_auc_is_half_when_the_groups_are_identical() -> None:
    same = [0.1, 0.2, 0.3, 0.4]
    assert auc_vs_reference(same, same) == pytest.approx(0.5)


def test_auc_is_one_when_every_value_is_larger() -> None:
    assert auc_vs_reference([5.0, 6.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert auc_vs_reference([1.0, 2.0], [5.0, 6.0]) == pytest.approx(0.0)


def test_auc_counts_ties_as_half_so_flat_bands_land_at_chance() -> None:
    """A band that is constant in both groups must not read as perfect separation."""
    assert auc_vs_reference([0.2] * 4, [0.2] * 4) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("auc", "expected"),
    [(0.90, "higher"), (0.65, "higher"), (0.58, "none"), (0.50, "none"),
     (0.42, "none"), (0.35, "lower"), (0.10, "lower")],
)
def test_direction_requires_clearing_the_margin(auc: float, expected: str) -> None:
    assert observed_direction(auc) == expected


def test_a_coherent_slowing_profile_is_reported_as_a_signature() -> None:
    healthy = _profiles({
        "delta": [0.20, 0.21, 0.22, 0.23], "theta": [0.10, 0.11, 0.12, 0.13],
        "alpha": [0.34, 0.35, 0.36, 0.37], "beta": [0.28, 0.29, 0.30, 0.31],
        "low_gamma": [0.03, 0.03, 0.03, 0.03],
        "theta_alpha_ratio": [0.30, 0.31, 0.32, 0.33],
    })
    affected = _profiles({
        "delta": [0.30, 0.31, 0.32, 0.33], "theta": [0.20, 0.21, 0.22, 0.23],
        "alpha": [0.20, 0.21, 0.22, 0.23], "beta": [0.18, 0.19, 0.20, 0.21],
        "low_gamma": [0.03, 0.03, 0.03, 0.03],
        "theta_alpha_ratio": [0.90, 0.91, 0.92, 0.93],
    })

    profile = build_condition_profile("PD", affected, healthy, BANDS)

    assert profile["has_signature"] is True
    assert CANONICAL_AXIS in profile["separating_bands"]
    assert profile["direction_vs_hc"]["alpha"] == "lower"
    assert profile["opposing_bands"] == []
    assert "low_gamma" not in profile["separating_bands"], "an unmoved band must not separate"


def test_no_signature_when_nothing_separates() -> None:
    identical = _profiles({band: [0.2, 0.25, 0.3, 0.35] for band in BANDS})
    profile = build_condition_profile("MS", identical, identical, BANDS)

    assert profile["has_signature"] is False
    assert profile["separating_bands"] == []
    assert "not conventional band power" in profile["note"]


def test_a_single_reversed_band_is_not_a_signature() -> None:
    """The MS case: less delta than controls, chance everywhere else.

    Lower slow-wave power is what a younger cohort looks like, not what an affected
    one looks like. One band moving the wrong way must not be dressed up as evidence.
    """
    healthy = _profiles({
        "delta": [0.26, 0.27, 0.28, 0.29], "theta": [0.13, 0.13, 0.14, 0.14],
        "alpha": [0.31, 0.31, 0.32, 0.32], "beta": [0.25, 0.25, 0.26, 0.26],
        "low_gamma": [0.03, 0.03, 0.03, 0.03],
        "theta_alpha_ratio": [0.40, 0.41, 0.42, 0.43],
    })
    younger = _profiles({
        "delta": [0.14, 0.15, 0.16, 0.17], "theta": [0.13, 0.14, 0.14, 0.13],
        "alpha": [0.32, 0.31, 0.32, 0.31], "beta": [0.26, 0.25, 0.26, 0.25],
        "low_gamma": [0.03, 0.03, 0.03, 0.03],
        "theta_alpha_ratio": [0.41, 0.42, 0.40, 0.43],
    })

    profile = build_condition_profile("MS", younger, healthy, BANDS)

    assert profile["has_signature"] is False, "one reversed band is not a signature"
    assert profile["separating_bands"] == ["delta"]
    assert profile["opposing_bands"] == ["delta"], "delta ran against the slowing direction"
    assert "opposite to the slowing direction" in profile["note"]
    assert "No pattern is claimed" in profile["note"]


def test_signature_needs_corroboration_not_just_the_canonical_axis() -> None:
    """theta/alpha alone, with every component band at chance, is not enough."""
    healthy = _profiles({
        "delta": [0.2, 0.2, 0.2, 0.2], "theta": [0.13, 0.13, 0.13, 0.13],
        "alpha": [0.32, 0.32, 0.32, 0.32], "beta": [0.25, 0.25, 0.25, 0.25],
        "low_gamma": [0.03, 0.03, 0.03, 0.03],
        "theta_alpha_ratio": [0.40, 0.41, 0.42, 0.43],
    })
    affected = _profiles({
        "delta": [0.2, 0.2, 0.2, 0.2], "theta": [0.13, 0.13, 0.13, 0.13],
        "alpha": [0.32, 0.32, 0.32, 0.32], "beta": [0.25, 0.25, 0.25, 0.25],
        "low_gamma": [0.03, 0.03, 0.03, 0.03],
        "theta_alpha_ratio": [0.90, 0.91, 0.92, 0.93],
    })

    profile = build_condition_profile("AD", affected, healthy, BANDS)
    assert profile["separating_bands"] == [CANONICAL_AXIS]
    assert profile["has_signature"] is False


def test_build_reference_excludes_controls_from_the_condition_map() -> None:
    groups = {name: _profiles({band: [0.2, 0.3] for band in BANDS})
              for name in ("HC", "AD", "PD", "MS")}
    reference = build_reference(groups, BANDS, "2026-01-01T00:00:00Z")

    assert set(reference["conditions"]) == {"AD", "PD", "MS"}
    assert reference["healthy"]["n"] == 2
    assert reference["canonical_axis"] == CANONICAL_AXIS


# --------------------------------------------------------------- installed bundle
@pytest.mark.skipif(not REFERENCE_PATH.exists(), reason="no band reference installed")
def test_installed_reference_still_declines_to_claim_an_ms_pattern() -> None:
    """Regression against the shipped cohort, not a fixture.

    If a future run makes MS report a signature, that is either a genuinely different
    cohort or a bug in the gate — either way it must not pass silently.
    """
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    conditions = reference["conditions"]

    assert conditions["MS"]["has_signature"] is False
    assert conditions["MS"]["opposing_bands"] == ["delta"]
    assert conditions["AD"]["has_signature"] is True
    assert conditions["PD"]["has_signature"] is True
    # PD is the strongest slowing profile in this cohort; AD is real but weaker.
    assert (conditions["PD"]["auc_vs_hc"][CANONICAL_AXIS]
            > conditions["AD"]["auc_vs_hc"][CANONICAL_AXIS])
