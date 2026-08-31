"""
vol_loader.py
=============
R26-DS-015 — eye/ms preprocessing

Standalone parser for Heidelberg Spectralis .vol files (HSF-OCT-1xx).

This implements the PUBLIC Heidelberg .vol binary format specification —
documented independently in multiple open-source readers (e.g.
github.com/ayl/heyexReader, the eyepy project's vol_reader.py, and
FabianRathke/octSegmentation's HDEVolImporter.m). It is infrastructure for
reading a vendor file format, not part of any MS-classification paper's
research contribution, so it is written standalone here rather than
depending on the (currently broken, due to an unrelated E2E-reader/
construct-typed version conflict) `eyepy` package.

Usage:
    from vol_loader import load_vol
    vol = load_vol("hc01_spectralis_macula_v1_s1_R.vol")
    vol.volume          # (n_bscans, size_y, size_x) float32, invalid px = nan
    vol.meta["scale_x"] # mm per pixel, B-scan width direction
    vol.meta["scale_y"] # mm per pixel, B-scan depth (axial) direction
    vol.meta["distance"]# mm between adjacent B-scans
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import construct as cs
import numpy as np

INVALID_PIXEL_THRESHOLD = 1.0e38  # sentinel used by the format for missing data


def _stack_ragged(arrays: list) -> np.ndarray:
    """Stack a list of 2D arrays that may have different row counts
    (e.g. vendor auto-segmentation num_seg can vary per B-scan within
    the same volume), padding shorter ones with nan. Same approach the
    eyepy library uses for this exact situation."""
    shapes = {a.shape for a in arrays}
    if len(shapes) == 1:
        return np.stack(arrays, axis=0)
    max_rows = max(s[0] for s in shapes)
    width = arrays[0].shape[1]
    padded = []
    for a in arrays:
        if a.shape[0] < max_rows:
            pad = np.full((max_rows - a.shape[0], width), np.nan, dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        padded.append(a)
    return np.stack(padded, axis=0)


def _get_date_adapter(construct_type, epoch, second_frac):
    class DateAdapter(cs.Adapter):
        def _decode(self, obj, context, path):
            return (epoch + timedelta(seconds=obj * second_frac)).isoformat()

    return DateAdapter(construct_type)


class _BscanAdapter(cs.Adapter):
    def _decode(self, obj, context, path):
        return np.ndarray(
            buffer=obj, dtype="float32", shape=(context._.size_y, context._.size_x)
        )


class _LocalizerAdapter(cs.Adapter):
    def _decode(self, obj, context, path):
        return np.ndarray(
            buffer=obj, dtype="uint8", shape=(context.size_y_slo, context.size_x_slo)
        )


class _SegmentationsAdapter(cs.Adapter):
    def _decode(self, obj, context, path):
        return np.ndarray(
            buffer=obj, dtype="float32", shape=(context.num_seg, context._.size_x)
        )


_IntDate = _get_date_adapter(cs.Int64ul, datetime(1601, 1, 1), 1e-7)
_FloatDate = _get_date_adapter(cs.Float64l, datetime(1899, 12, 30), 60 * 60 * 24)
_Localizer = _LocalizerAdapter(cs.Bytes(cs.this.size_x_slo * cs.this.size_y_slo))
_Segmentations = _SegmentationsAdapter(
    cs.Bytes(cs.this.num_seg * cs.this._.size_x * 4)
)
_Bscan = _BscanAdapter(cs.Bytes(cs.this._.size_y * cs.this._.size_x * 4))

_bscan_format = cs.Struct(
    "version" / cs.PaddedString(12, "ascii"),
    "bscan_hdr_size" / cs.Int32ul,
    "start_x" / cs.Float64l,
    "start_y" / cs.Float64l,
    "end_x" / cs.Float64l,
    "end_y" / cs.Float64l,
    "num_seg" / cs.Int32ul,
    "seg_offset" / cs.Int32ul,
    "quality" / cs.Float32l,
    "shift" / cs.Int32ul,
    "iv_transformation" / cs.Float32l[6],
    "__empty" / cs.Padding(168),
    "layer_segmentations" / _Segmentations,
    "__empty2"
    / cs.Padding(cs.this.bscan_hdr_size - 256 - cs.this.num_seg * 4 * cs.this._.size_x),
    "data" / _Bscan,
)

_vol_format = cs.Struct(
    "version" / cs.PaddedString(12, "ascii"),
    "size_x" / cs.Int32ul,
    "n_bscans" / cs.Int32ul,
    "size_y" / cs.Int32ul,
    "scale_x" / cs.Float64l,
    "distance" / cs.Float64l,
    "scale_y" / cs.Float64l,
    "size_x_slo" / cs.Int32ul,
    "size_y_slo" / cs.Int32ul,
    "scale_x_slo" / cs.Float64l,
    "scale_y_slo" / cs.Float64l,
    "field_size_slo" / cs.Int32ul,
    "scan_focus" / cs.Float64l,
    "scan_position" / cs.PaddedString(4, "ascii"),
    "exam_time" / _IntDate,
    "scan_pattern" / cs.Int32sl,
    "bscan_hdr_size" / cs.Int32ul,
    "id" / cs.PaddedString(16, "ascii"),
    "reference_id" / cs.PaddedString(16, "ascii"),
    "pid" / cs.Int32ul,
    "patient_id" / cs.PaddedString(24, "ascii"),
    "dob" / _FloatDate,
    "vid" / cs.Int32ul,
    "visit_id" / cs.PaddedString(24, "ascii"),
    "visit_date" / _FloatDate,
    "grid_type" / cs.Int32ul,
    "grid_offset" / cs.Int32ul,
    "grid_type1" / cs.Int32ul,
    "grid_offset1" / cs.Int32ul,
    "prog_id" / cs.PaddedString(34, "ascii"),
    "__empty" / cs.Padding(1790),
    "localizer" / _Localizer,
    "bscans" / _bscan_format[cs.this.n_bscans],
)


@dataclass
class VolData:
    volume: np.ndarray          # (n_bscans, size_y, size_x) float32, invalid px -> nan
    vendor_layers: np.ndarray   # (n_bscans, num_seg, size_x) float32, vendor auto-seg (may be empty)
    localizer: np.ndarray       # (size_y_slo, size_x_slo) uint8 SLO fundus image
    meta: dict


def load_vol(path: str) -> VolData:
    """Parse a Heidelberg .vol file into a volume array + metadata."""
    with open(path, "rb") as f:
        parsed = _vol_format.parse_stream(f)

    volume = _stack_ragged([b.data for b in parsed.bscans])
    volume = np.where(volume >= INVALID_PIXEL_THRESHOLD, np.nan, volume)

    vendor_layers = _stack_ragged([b.layer_segmentations for b in parsed.bscans])
    vendor_layers = np.where(vendor_layers >= INVALID_PIXEL_THRESHOLD, np.nan, vendor_layers)

    meta = {
        "version": parsed.version.strip(),
        "size_x": parsed.size_x,
        "size_y": parsed.size_y,
        "n_bscans": parsed.n_bscans,
        "scale_x": parsed.scale_x,     # mm/px, B-scan width (A-scan spacing)
        "scale_y": parsed.scale_y,     # mm/px, B-scan depth (axial)
        "distance": parsed.distance,   # mm between B-scans
        "scan_position": parsed.scan_position.strip(),  # 'OD' or 'OS'
        "scan_pattern": parsed.scan_pattern,
        "patient_id": parsed.patient_id.strip(),
        "visit_id": parsed.visit_id.strip(),
    }

    return VolData(
        volume=volume,
        vendor_layers=vendor_layers,
        localizer=parsed.localizer,
        meta=meta,
    )


if __name__ == "__main__":
    import sys

    vol = load_vol(sys.argv[1])
    print(f"volume shape: {vol.volume.shape}")
    print(f"meta: {vol.meta}")
    print(f"vendor_layers shape: {vol.vendor_layers.shape}")