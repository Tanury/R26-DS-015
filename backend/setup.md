# Modal Notebook Setup For `eeg_ad_hc_phase1`

This guide explains how to upload and run this project in a Modal Notebook using an `L40S` GPU.

References:

- Modal docs: <https://modal.com/docs>
- Modal Notebooks: <https://modal.com/docs/guide/notebooks#modal-notebooks>
- Modal GPU guide: <https://modal.com/docs/guide/gpu>
- Modal CPU, memory, and disk guide: <https://modal.com/docs/guide/resources>
- Modal Volumes guide: <https://modal.com/docs/guide/volumes>

## 1. Recommended Modal Resources

For this Phase 1 EEGNet baseline:

| Resource | Recommended | Why |
|---|---:|---|
| GPU | `L40S` | Good performance headroom; Modal supports `L40S`, and its GPU guide notes it has 48 GB GPU RAM. |
| CPU cores | `4` cores | Enough for MNE preprocessing, loading, filtering, and PyTorch dataloading. |
| RAM | `16 GB` minimum, `32 GB` safer | Dataset is small, but EEG preprocessing can spike memory when loading raw recordings. |
| Ephemeral disk | Default is usually fine; request `50-100 GB` only if needed | Store important data in Volumes, not only on notebook disk. |
| Persistent storage | Modal Volumes | Files under attached `/mnt/...` Volumes persist across kernel restarts. Files outside Volumes can disappear when the kernel stops. |

Recommended starting profile in the Modal Notebook sidebar:

```text
GPU: L40S
CPU: 4 physical cores
Memory: 32 GB
Idle timeout: 30-60 minutes while actively working
```

Cost note: Modal bills notebooks while the kernel is running. Stop the kernel when finished.

## 2. Is Storage Permanent?

Yes, if you save files inside an attached Modal Volume.

No, if you save files only in the notebook container filesystem such as `/root` and then stop the kernel.

Use this rule:

```text
/root/eeg_ad_hc_phase1/          -> code, can be re-uploaded/re-cloned
/mnt/eeg-dataset-vol/...         -> raw dataset and processed EEG tensors, persistent
/mnt/eeg-artifacts-vol/...       -> checkpoints, embeddings, metrics, figures, reports, persistent
```

Modal’s notebook docs say attached Volumes appear under `/mnt` and files saved there persist across kernel startups. The same page warns that the notebook container is ephemeral, so anything outside an attached Volume can disappear when the kernel shuts down.

## 3. Create Modal Volumes

You can create Volumes in the Modal dashboard from the Notebook filesystem sidebar, or from a local terminal after installing Modal:

```bash
pip install modal
modal setup
modal volume create eeg-dataset-vol
modal volume create eeg-artifacts-vol
```

Attach both Volumes to the Modal Notebook. They should appear as:

```text
/mnt/eeg-dataset-vol
/mnt/eeg-artifacts-vol
```

Suggested Volume contents:

```text
/mnt/eeg-dataset-vol/
├── raw/openneuro_ds004504/
└── processed/
    ├── epochs/
    └── preprocessing_logs/

/mnt/eeg-artifacts-vol/
├── checkpoints/
├── final/
├── embeddings/
├── figures/
├── reports/
└── logs/
```

## 4. Upload This Project To Modal Notebook

### Option A: Upload ZIP Through Modal Files Panel

1. Zip this local folder:

```text
F:\eeg_ad_hc_phase1
```

2. Open <https://modal.com/notebooks>.
3. Create a new notebook.
4. Set compute resources in the sidebar:

```text
GPU: L40S
CPU: 4
Memory: 32 GB
```

5. Attach the two Volumes:

```text
eeg-dataset-vol
eeg-artifacts-vol
```

6. Upload the ZIP using the Modal file viewer.
7. In a notebook cell, unzip it:

```bash
!unzip eeg_ad_hc_phase1.zip -d /root/
!ls /root/eeg_ad_hc_phase1
```

### Option B: Upload Files Directly

Use the file viewer upload button and place the project at:

```text
/root/eeg_ad_hc_phase1
```

This is fine for code. Do not store the dataset only under `/root`.

### Option C: Git Clone

If you later push this project to GitHub:

```bash
%cd /root
!git clone <YOUR_REPO_URL> eeg_ad_hc_phase1
%cd /root/eeg_ad_hc_phase1
```

## 5. Install Dependencies In The Notebook

Run:

```bash
%cd /root/eeg_ad_hc_phase1
!pip install -r requirements.txt
```

If PyTorch is already installed in Modal’s default image, this may be quick. If CUDA packages are upgraded, restart the kernel after installation.

Check GPU:

```bash
!nvidia-smi
```

Check Python imports:

```bash
!python -c "import torch, mne, pandas, sklearn, gradio; print('imports ok', torch.cuda.is_available())"
```

## 6. Configure Paths For Modal Volumes

Run from the project directory in your Modal Notebook:

```bash
%cd /root/eeg_ad_hc_phase1
```

Then run this cell to update the config files for Modal Volumes:

```python
from pathlib import Path
import yaml

modal_paths = {
    "dataset_root": "/mnt/eeg-dataset-vol/raw/openneuro_ds004504",
    "labels_dir": "/mnt/eeg-artifacts-vol/labels",
    "splits_dir": "/mnt/eeg-artifacts-vol/splits",
    "processed_epochs_dir": "/mnt/eeg-dataset-vol/processed/epochs",
    "preprocessing_logs_dir": "/mnt/eeg-dataset-vol/processed/preprocessing_logs",
    "checkpoints_dir": "/mnt/eeg-artifacts-vol/checkpoints",
    "final_model_dir": "/mnt/eeg-artifacts-vol/final",
    "embeddings_dir": "/mnt/eeg-artifacts-vol/embeddings",
    "figures_dir": "/mnt/eeg-artifacts-vol/figures",
    "reports_dir": "/mnt/eeg-artifacts-vol/reports",
    "logs_dir": "/mnt/eeg-artifacts-vol/logs",
}

for config_name in ["configs/preprocessing.yaml", "configs/training.yaml"]:
    config_path = Path(config_name)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg["paths"] = dict(modal_paths)
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Updated {config_name}")
```

Important: `configs/training.yaml` inherits from `configs/preprocessing.yaml`, but it also overrides `paths`. Therefore, the `training.yaml` `paths` block must include the full path set, not only checkpoint/report paths.

`configs/training_synthetic_test.yaml` inherits from `configs/training.yaml`, so it receives the same Modal paths automatically. Use it only for smoke testing, not final Phase 1 metrics.

Create the folders:

```bash
!mkdir -p /mnt/eeg-dataset-vol/raw/openneuro_ds004504
!mkdir -p /mnt/eeg-dataset-vol/processed/epochs
!mkdir -p /mnt/eeg-dataset-vol/processed/preprocessing_logs
!mkdir -p /mnt/eeg-artifacts-vol/labels /mnt/eeg-artifacts-vol/splits
!mkdir -p /mnt/eeg-artifacts-vol/checkpoints /mnt/eeg-artifacts-vol/final
!mkdir -p /mnt/eeg-artifacts-vol/embeddings /mnt/eeg-artifacts-vol/figures
!mkdir -p /mnt/eeg-artifacts-vol/reports /mnt/eeg-artifacts-vol/logs
```

Verify the resolved config paths:

```bash
!python - <<'PY'
from src.utils.config_loader import load_config

required_path_keys = [
    "dataset_root",
    "labels_dir",
    "splits_dir",
    "processed_epochs_dir",
    "preprocessing_logs_dir",
    "checkpoints_dir",
    "figures_dir",
    "reports_dir",
    "logs_dir",
]

for config_path in ["configs/training.yaml", "configs/training_synthetic_test.yaml"]:
    cfg = load_config(config_path)
    print(f"\n{config_path}")

    for key in required_path_keys:
        value = cfg["paths"][key]
        print(f"  {key}: {value}")

    assert cfg["paths"]["dataset_root"].startswith("/mnt/eeg-dataset-vol")
    assert cfg["paths"]["checkpoints_dir"].startswith("/mnt/eeg-artifacts-vol")
    assert cfg["preprocessing"]["low_freq_hz"] == 0.5
    assert cfg["preprocessing"]["high_freq_hz"] == 40.0
    assert len(cfg["preprocessing"]["selected_channels"]) == 19

print("\nModal config paths verified.")
PY
```

## 7. Download OpenNeuro `ds004504`

Recommended:

```bash
!pip install -q --upgrade openneuro-py
!openneuro-py download --dataset ds004504 --target-dir /mnt/eeg-dataset-vol/raw/openneuro_ds004504
```

Alternative GitHub clone:

```bash
!git clone https://github.com/OpenNeuroDatasets/ds004504.git /mnt/eeg-dataset-vol/raw/openneuro_ds004504
```

Expected dataset facts:

```text
Subjects: 88
AD: 36
HC/CN: 29
FTD: 23 excluded from Phase 1
Channels: 19
Sampling rate: 500 Hz
Approximate size: 2.6 GB
```

## 8. Run The Phase 1 Pipeline

Always run from the project directory:

```bash
%cd /root/eeg_ad_hc_phase1
```

Prepare labels:

```bash
!python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

Create leakage-safe subject splits:

```bash
!python -m src.data.create_subject_splits --config configs/training.yaml
!python -m src.evaluation.leakage_check --config configs/training.yaml
```

Preprocess all AD/HC subjects:

```bash
!python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all
```

If preprocessing settings changed, force-refresh processed files:

```bash
!python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
```

Validate preprocessed channel order and tensor shapes:

```bash
!python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
```

Train EEGNet:

```bash
!python -m src.training.train_eegnet --config configs/training.yaml
```

Evaluate subject-level metrics:

```bash
!python -m src.evaluation.evaluate_subject_level --config configs/training.yaml
```

Generate one subject JSON report:

```bash
!python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-001
```

Run the Modal post-training smoke test:

```bash
!python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --fold 0
```

If you trained all five folds, run:

```bash
!python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --all-folds
```

Run tests:

```bash
!pytest -q
```

## 9. Run Gradio In Modal Notebook

Start the app:

```bash
!python -m app.gradio_app
```

If Modal Notebook does not expose the local Gradio URL cleanly, use Gradio sharing temporarily by editing `app/gradio_app.py`:

```python
build_app().launch(share=True)
```

Then run again:

```bash
!python -m app.gradio_app
```

Use `share=True` only for demos, because it creates a public temporary link.

## 10. What To Download For Submission

From `/mnt/eeg-artifacts-vol`, download:

```text
labels/
splits/
reports/phase1_metrics.json
reports/modal_post_train_check.json
reports/sub-001_prediction_report.json
figures/confusion_matrix.png
figures/roc_curve.png
figures/sub-001_probability_bar_chart.png
figures/sub-001_epoch_probability_distribution.png
checkpoints/eegnet_fold0_best.pth
embeddings/sub-001_z_eeg.npy
```

Also keep:

```text
guide.md
setup.md
README.md
configs/
src/
app/
notebooks/
tests/
```

## 11. Practical Settings

For fast debugging:

```yaml
training:
  epochs: 2
  batch_size: 16
```

For final Phase 1 run:

```yaml
training:
  epochs: 20
  batch_size: 16
  early_stopping_patience: 5
```

If memory becomes tight:

- Lower `batch_size` from `16` to `8`.
- Keep processed tensors in `/mnt/eeg-dataset-vol/processed`.
- Stop other notebooks using the same GPU.

If preprocessing is slow:

- Keep CPU at `4`.
- Increase to `8` only if filtering/resampling becomes the bottleneck.
- Keep `num_workers: 0` first in notebooks; increase later if stable.

## 12. Final Checklist

- [ ] Notebook uses `L40S`.
- [ ] CPU set to `4`.
- [ ] Memory set to `32 GB`.
- [ ] Dataset Volume attached at `/mnt/eeg-dataset-vol`.
- [ ] Artifacts Volume attached at `/mnt/eeg-artifacts-vol`.
- [ ] Dataset downloaded under `/mnt/eeg-dataset-vol/raw/openneuro_ds004504`.
- [ ] Config paths point to `/mnt`.
- [ ] Labels prepared.
- [ ] FTD excluded.
- [ ] Subject splits created.
- [ ] Leakage check passed.
- [ ] EEGNet trained.
- [ ] Subject-level evaluation generated.
- [ ] JSON report generated.
- [ ] Modal post-training smoke test passed.
- [ ] Gradio demo tested.
- [ ] Kernel stopped after use to avoid extra cost.
