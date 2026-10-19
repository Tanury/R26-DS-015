"""
run_oct5k_specificity.py
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from thickness import compute_thickness_features, central_bscan_window

BAND_NAMES = ["ILM_OPL", "OPL_ISOS", "ISOS_IBRPE", "IBRPE_OBRPE"]
BOUNDARY_COLS = ["ILM", "OPL", "IS-OS", "IBRPE", "OBRPE"]  # exact CSV header order

# See CAVEAT ON SCALE above -- override with --scale-y-um if you confirm
# the real figure from Rasti et al.'s device documentation.
SCALE_Y_UM_DEFAULT = 3.5

_IMG_NUM_RE = re.compile(r"Image\s+(\d+)\.csv$", re.IGNORECASE)


def _group_from_folder(folder_name: str) -> str:
    # "AMD Part1" / "AMD Part2" -> "AMD"; "DME" -> "DME"; "Normal Part1" -> "Normal"
    return folder_name.split(" Part")[0]


def _load_volume_boundaries(volume_dir: Path) -> np.ndarray:
    """volume_dir is e.g. '.../AMD (1).E2E'. Returns (n_bscans, 5, size_x)."""
    timestamp_dirs = [d for d in volume_dir.iterdir() if d.is_dir()]
    if len(timestamp_dirs) != 1:
        raise ValueError(f"{volume_dir} -- expected exactly 1 timestamp subfolder, found {len(timestamp_dirs)}")
    scan_dir = timestamp_dirs[0]

    csv_files = []
    for f in scan_dir.glob("Image *.csv"):
        m = _IMG_NUM_RE.search(f.name)
        if m:
            csv_files.append((int(m.group(1)), f))
    csv_files.sort(key=lambda t: t[0])
    if not csv_files:
        raise ValueError(f"{scan_dir} -- no 'Image N.csv' files found")

    bscans = []
    size_x = None
    for _, f in csv_files:
        df = pd.read_csv(f)
        if list(df.columns) != ["x"] + BOUNDARY_COLS:
            raise ValueError(f"{f} -- unexpected columns {list(df.columns)}, expected ['x'] + {BOUNDARY_COLS}")
        df = df.sort_values("x")
        if size_x is None:
            size_x = len(df)
        elif len(df) != size_x:
            raise ValueError(f"{f} -- {len(df)} rows, expected {size_x} (inconsistent with earlier B-scans in this volume)")
        bscans.append(df[BOUNDARY_COLS].values.T)  # (5, size_x)

    return np.stack(bscans, axis=0).astype(np.float64)  # (n_bscans, 5, size_x)


def run_specificity(oct5k_root: str, model_path: str, out_csv: str,
                     grading: int = 1, scale_y_um: float = SCALE_Y_UM_DEFAULT,
                     central_window: int = 10) -> pd.DataFrame:
    root = Path(oct5k_root)
    manual_dir = root / "Boundaries" / "Boundaries_Manual" / f"Grading_{grading}"
    if not manual_dir.exists():
        raise FileNotFoundError(f"{manual_dir} not found -- check oct5k_root and --grading")

    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]
    print(f"Loaded comparison model ({bundle['model_type']}, LOOCV acc={bundle['loocv_accuracy']:.3f}, "
          f"trained on {bundle['trained_on_n']} subjects)\n")
    print(f"Using Grading_{grading} annotations, scale_y={scale_y_um} um/px "
          f"({'default, OCR-derived -- verify' if scale_y_um == SCALE_Y_UM_DEFAULT else 'user override'})\n")

    scale_y_mm = scale_y_um / 1000.0

    group_folders = sorted(d for d in manual_dir.iterdir() if d.is_dir())
    rows = []
    failed = []

    for group_folder in group_folders:
        disease_group = _group_from_folder(group_folder.name)
        volume_dirs = sorted(d for d in group_folder.iterdir() if d.is_dir())
        for volume_dir in volume_dirs:
            subject_id = volume_dir.name.replace(".E2E", "")
            try:
                boundaries = _load_volume_boundaries(volume_dir)
                n_bscans = boundaries.shape[0]
                window = central_bscan_window(n_bscans, window=central_window)
                feats = compute_thickness_features(boundaries, scale_y_mm=scale_y_mm, bscan_indices=window)
            except Exception as e:
                failed.append(f"[{subject_id}] {e}")
                continue

            X = feats.subject_vector.reshape(1, -1)  # (1, 4), order = BAND_NAMES
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            proba_ms = model.predict_proba(X_scaled)[0, 1]

            rows.append({
                "subject_id": subject_id,
                "disease_group": disease_group,
                "is_disease_lookalike": disease_group in ("AMD", "DME"),
                "expected_label": "HC",  # see METHODOLOGY above -- always HC
                "predicted_label": "MS" if pred == 1 else "HC",
                "p_ms": float(proba_ms),
                "false_positive": pred == 1,
                "n_bscans": n_bscans,
                **{f"band_{name}_um": val for name, val in zip(BAND_NAMES, feats.subject_vector)},
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Processed {len(df)} subjects ({len(failed)} failed)\n")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")
        print()

    print("=" * 80)
    print("SPECIFICITY (expected label = HC for everyone -- see methodology note)")
    print("=" * 80)
    for group in ["AMD", "DME", "Normal"]:
        sub = df[df["disease_group"] == group]
        if len(sub) == 0:
            continue
        fp_rate = sub["false_positive"].mean()
        label = "look-alike pathology" if group != "Normal" else "true healthy control (sanity check, not a specificity test)"
        print(f"  {group:8s} (n={len(sub):2d}, {label}): "
              f"specificity={1 - fp_rate:.3f}  false-positive rate={fp_rate:.3f}  "
              f"mean p(MS)={sub['p_ms'].mean():.3f}")

    lookalike = df[df["is_disease_lookalike"]]
    if len(lookalike) > 0:
        overall_fp = lookalike["false_positive"].mean()
        print(f"\n  OVERALL AMD+DME specificity (the headline number): "
              f"{1 - overall_fp:.3f}  (n={len(lookalike)}, false-positive rate={overall_fp:.3f})")

    print(f"\nSaved per-subject results to: {out_csv}")


    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("oct5k_root", help="Path to the OCT5k folder (contains Boundaries/, Masks/, etc.)")
    parser.add_argument("model_path", help="Path to 4band_comparison_model.joblib from train_4band_comparison_model.py")
    parser.add_argument("--out", default="oct5k_specificity_results.csv")
    parser.add_argument("--grading", type=int, default=1, choices=[1, 2, 3],
                         help="Which of OCT5k's 3 graders' annotations to use (default: 1)")
    parser.add_argument("--scale-y-um", type=float, default=SCALE_Y_UM_DEFAULT,
                         help="Axial pixel scale in microns/pixel -- override if you confirm the real device spec")
    parser.add_argument("--central-window", type=int, default=10)
    args = parser.parse_args()
    run_specificity(args.oct5k_root, args.model_path, args.out,
                     grading=args.grading, scale_y_um=args.scale_y_um,
                     central_window=args.central_window)