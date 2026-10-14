"""
MRI preprocessing pipeline for the API.
Step order: DICOM → NIfTI → Skull Strip → Register → Bias Correct

"""

import os
import io
import base64
import subprocess
import tempfile
import zipfile
import time
import numpy as np
from pathlib import Path


# ── FSL paths ─────────────────────────────────────────────────────────────────
FSLDIR        = os.environ.get("FSLDIR", "/Users/tanuridissanayaka/fsl")
ATLAS         = f"{FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz"
PROJECT_ATLAS = "shared/atlas/MNI152_T1_1mm.nii.gz"


def _atlas_path() -> str:
    if os.path.exists(ATLAS):
        return ATLAS
    if os.path.exists(PROJECT_ATLAS):
        return PROJECT_ATLAS
    raise FileNotFoundError(
        f"MNI152 atlas not found at {ATLAS} or {PROJECT_ATLAS}."
    )


# ── Slice extraction ──────────────────────────────────────────────────────────

def _extract_slices(nifti_path: str, label: str) -> dict:
    try:
        import nibabel as nib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img  = nib.load(nifti_path)
        data = np.asanyarray(img.dataobj, dtype=np.float32)

        # Handle 4D
        if data.ndim == 4:
            data = data[..., 0]

        if data.max() <= 0:
            return {v: _placeholder_b64(v, "empty volume") for v in ["axial", "coronal", "sagittal"]}

        # Robust normalisation — use non-zero voxels only
        nonzero = data[data > 0]
        p1, p99 = np.percentile(nonzero, [1, 99])
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-8)

        slices = {}
        cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2

        views = {
            "axial":    np.rot90(data[:, :, cz]),
            "coronal":  np.rot90(data[:, cy, :]),
            "sagittal": np.rot90(data[cx, :, :]),
        }

        for key, sl in views.items():
            fig, ax = plt.subplots(figsize=(3, 3), facecolor="#0a0a0a")
            ax.imshow(sl, cmap="gray", aspect="auto",
                      interpolation="bilinear", vmin=0, vmax=1)
            ax.axis("off")
            fig.tight_layout(pad=0.2)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, facecolor="#0a0a0a",
                        bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            slices[key] = base64.b64encode(buf.read()).decode("utf-8")

        return slices

    except Exception as e:
        return {v: _placeholder_b64(v, str(e)[:40]) for v in ["axial", "coronal", "sagittal"]}


def _placeholder_b64(view: str, msg: str = "") -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="#111")
        ax.text(0.5, 0.5, f"{view}\n{msg[:40]}", ha="center", va="center",
                color="#555", fontsize=7, transform=ax.transAxes, wrap=True)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor="#111")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list, label: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}): "
            f"stderr: {result.stderr[-400:]}"
        )


def _volume_max(path: str) -> float:
    try:
        import nibabel as nib
        data = nib.load(path).get_fdata()
        return float(data.max())
    except Exception:
        return 0.0


def _step_result(step_id, name, description, nifti_path, elapsed, error=None):
    return {
        "step":        step_id,
        "name":        name,
        "description": description,
        "elapsed_s":   round(elapsed, 1),
        "success":     error is None,
        "error":       error,
        "slices":      (
            _extract_slices(nifti_path, name) if error is None
            else {v: _placeholder_b64(v, error[:40] if error else "")
                  for v in ["axial", "coronal", "sagittal"]}
        ),
        "nifti_path":  nifti_path if error is None else None,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(zip_bytes: bytes, filename: str) -> dict:
    """
    Correct pipeline order:
        0. DICOM → NIfTI + reorient
        1. Skull Strip (BET on raw scan — most reliable input for BET)
        2. Affine Registration (register skull-stripped brain to MNI)
        3. N4 Bias Correction (on registered brain)
    """
    workdir    = tempfile.mkdtemp(prefix="r26_pipeline_")
    total_start = time.time()
    steps      = []

    try:
        # ── Unzip ──────────────────────────────────────────────────────────
        dicom_dir = Path(workdir) / "dicom"
        dicom_dir.mkdir()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(str(dicom_dir))

        dcm_files = list(dicom_dir.rglob("*.dcm"))
        if not dcm_files:
            dcm_files = [f for f in dicom_dir.rglob("*")
                         if f.is_file() and not f.suffix]
        if not dcm_files:
            raise FileNotFoundError("No DICOM files found in zip.")

        actual_dicom_dir = str(dcm_files[0].parent)
        subject_id = filename.replace(".zip", "").replace(" ", "_")[:28]

        # ────────────────────────────────────────────────────────────────────
        # STEP 0 — DICOM → NIfTI + reorient to standard space
        # ────────────────────────────────────────────────────────────────────
        step_start = time.time()
        nifti_dir  = Path(workdir) / "nifti"
        nifti_dir.mkdir()

        try:
            _run([
                "dcm2niix", "-z", "y", "-f", subject_id,
                "-o", str(nifti_dir), "-b", "n", actual_dicom_dir,
            ], "dcm2niix")

            nifti_files = (list(nifti_dir.glob("*.nii.gz"))
                           + list(nifti_dir.glob("*.nii")))
            if not nifti_files:
                raise FileNotFoundError("dcm2niix produced no NIfTI output")

            raw_nifti  = str(nifti_files[0])
            reoriented = str(nifti_dir / f"{subject_id}_std.nii.gz")
            _run(["fslreorient2std", raw_nifti, reoriented], "fslreorient2std")

            steps.append(_step_result(
                0, "DICOM → NIfTI",
                "Converted DICOM to NIfTI format and reoriented to standard space",
                reoriented, time.time() - step_start
            ))

        except Exception as e:
            steps.append(_step_result(
                0, "DICOM → NIfTI",
                "Converted DICOM to NIfTI format",
                workdir, time.time() - step_start, str(e)
            ))
            raise

        # ────────────────────────────────────────────────────────────────────
        # STEP 1 — Skull Stripping on raw reoriented scan
        # Run BET on the raw scan BEFORE registration.
        # BET is most reliable on native-space T1 images.
        # ────────────────────────────────────────────────────────────────────
        step_start = time.time()
        skull_dir  = Path(workdir) / "skull_stripped"
        skull_dir.mkdir()
        brain_out  = str(skull_dir / f"{subject_id}_brain.nii.gz")

        try:
            # Try progressively more conservative thresholds
            # -f 0.5 = standard, removes more; -f 0.3 = conservative, keeps more brain
            bet_succeeded = False
            for f_thresh in ["0.5", "0.4", "0.3"]:
                try:
                    _run([
                        "bet", reoriented, brain_out,
                        "-f", f_thresh,
                        "-R",   # robust — iterative re-estimation of brain centre
                        "-S",   # remove eyes and optic nerve  
                        "-B",   # bias field and neck cleanup
                    ], f"bet -f {f_thresh}")
                    if Path(brain_out).exists() and _volume_max(brain_out) > 0:
                        bet_succeeded = True
                        break
                except RuntimeError:
                    continue

            if not bet_succeeded:
                # Last resort — minimal flags
                _run(["bet", reoriented, brain_out, "-f", "0.4"], "bet minimal")

            if not Path(brain_out).exists():
                raise FileNotFoundError("BET produced no output file")

            if _volume_max(brain_out) == 0:
                raise RuntimeError("BET output is empty — skull stripping failed")

            steps.append(_step_result(
                1, "Skull Stripping",
                "Removed skull, eyes, and non-brain tissue using FSL BET",
                brain_out, time.time() - step_start
            ))

        except Exception as e:
            steps.append(_step_result(
                1, "Skull Stripping",
                "Removed non-brain tissue using FSL BET",
                reoriented, time.time() - step_start, str(e)
            ))
            raise

        # ────────────────────────────────────────────────────────────────────
        # STEP 2 — Affine Registration to MNI152
        # Register the skull-stripped brain (not raw scan) to MNI space.
        # Using mutualinfo cost — more robust for skull-stripped inputs.
        # ────────────────────────────────────────────────────────────────────
        step_start = time.time()
        reg_dir    = Path(workdir) / "registered"
        reg_dir.mkdir()
        registered = str(reg_dir / f"{subject_id}_reg.nii.gz")

        try:
            _run([
                "flirt",
                "-in",      brain_out,
                "-ref",     _atlas_path(),
                "-out",     registered,
                "-dof",     "12",
                "-interp",  "trilinear",
                "-cost",    "normmi",       # normalised mutual info — best for brain-only
                "-searchrx", "-90", "90",
                "-searchry", "-90", "90",
                "-searchrz", "-90", "90",
                "-searchcost", "normmi",
            ], "flirt")

            reg_max = _volume_max(registered)
            if reg_max < 1.0:
                # Registration failed silently — use skull-stripped brain directly
                raise RuntimeError(
                    f"FLIRT produced near-empty output (max={reg_max:.2f}). "
                    "Using skull-stripped brain as fallback."
                )

            steps.append(_step_result(
                2, "Affine Registration",
                "Registered skull-stripped brain to MNI152 standard space using FSL FLIRT",
                registered, time.time() - step_start
            ))

        except Exception as e:
            # FALLBACK: if registration fails, use skull-stripped brain directly
            # This is still valid for inference — just not in MNI space
            import shutil
            shutil.copy(brain_out, registered)
            steps.append(_step_result(
                2, "Affine Registration",
                "Registration attempted; using skull-stripped brain (fallback)",
                registered, time.time() - step_start,
                f"FLIRT issue — using skull-stripped brain: {str(e)[:80]}"
            ))
            # Don't raise — continue to bias correction with fallback

        # ────────────────────────────────────────────────────────────────────
        # STEP 3 — N4 Bias Field Correction
        # ────────────────────────────────────────────────────────────────────
        step_start  = time.time()
        denoise_dir = Path(workdir) / "denoised"
        denoise_dir.mkdir()
        denoised    = str(denoise_dir / f"{subject_id}_n4.nii.gz")

        try:
            import ants

            img  = ants.image_read(registered)

            # Ensure we have a valid 3D image
            if img.dimension != 3:
                raise ValueError(f"Expected 3D ANTs image, got {img.dimension}D")

            mask = ants.get_mask(img, low_thresh=0.01, cleanup=2)
            corrected = ants.n4_bias_field_correction(
                img, mask=mask,
                shrink_factor=4,
                convergence={"iters": [50, 50, 50, 50], "tol": 0.0001},
                spline_param=200,
                return_bias_field=False,
                verbose=False,
            )
            ants.image_write(corrected, denoised)

            if _volume_max(denoised) == 0:
                raise RuntimeError("N4 output is empty")

            steps.append(_step_result(
                3, "N4 Bias Correction",
                "Removed MRI intensity non-uniformity using ANTs N4 bias field correction",
                denoised, time.time() - step_start
            ))

        except Exception as e:
            import shutil
            shutil.copy(registered, denoised)
            steps.append(_step_result(
                3, "N4 Bias Correction",
                "N4 bias correction (used registered brain as fallback)",
                denoised, time.time() - step_start,
                f"ANTs N4 issue — using registered brain: {str(e)[:80]}"
            ))
            # Don't raise — use fallback for inference

        return {
            "steps":           steps,
            "final_nifti":     denoised,
            "workdir":         workdir,
            "subject_id":      subject_id,
            "total_elapsed_s": round(time.time() - total_start, 1),
            "success":         True,
            "error":           None,
        }

    except Exception as e:
        return {
            "steps":           steps,
            "final_nifti":     None,
            "workdir":         workdir,
            "subject_id":      filename,
            "total_elapsed_s": round(time.time() - total_start, 1),
            "success":         False,
            "error":           str(e),
        }