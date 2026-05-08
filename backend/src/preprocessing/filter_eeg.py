from __future__ import annotations


def filter_raw(raw, low_freq_hz: float, high_freq_hz: float):
    return raw.copy().filter(low_freq_hz, high_freq_hz, fir_design="firwin", verbose=False)
