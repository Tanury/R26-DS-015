# EEG AD/HC Phase 1 Guide

## 1. Project Goal

This repository implements a Phase 1 research prototype for EEG Signal Representation Learning using OpenNeuro `ds004504`.

It is limited to:

- OpenNeuro `ds004504`
- Resting-state closed-eyes EEG
- AD vs Healthy Control binary classification
- FTD exclusion
- EEG preprocessing
- EEGNet baseline training
- Subject-level evaluation
- 256D L2-normalized `z_eeg`
- Gradio research demo

Use safe wording: the model predicts `Alzheimer's EEG Pattern`, not a clinical diagnosis.

This guide defaults to local execution from the project folder. For Modal Notebook upload, persistent storage, and `L40S` resource setup, see [setup.md](setup.md).

## 2. Local Machine Quick Start

Run these commands from the project folder:

```text
C:\Users\skspa\Downloads\cookie\eeg_ad_hc_phase1_fixed\eeg_ad_hc_phase1_fixed
```

Windows PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Download Dataset command:

```powershell
pip install -q --upgrade openneuro-py
openneuro-py download --dataset ds004504 --target-dir data/raw/openneuro_ds004504
```


Local output folders:

```text
data/raw/openneuro_ds004504/          raw OpenNeuro dataset
data/labels/                         generated labels
data/splits/                         subject-level folds
data/processed/epochs/               preprocessed epoch tensors
data/processed/preprocessing_logs/    preprocessing logs
models/checkpoints/                  trained checkpoints
models/embeddings/                   z_eeg embeddings
outputs/reports/                     metrics and JSON reports
outputs/figures/                     plots
outputs/gradio_cache/                Gradio-display copies
```

Full local run order:

```powershell
python -m src.data.prepare_labels --config configs/preprocessing.yaml
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
python -m src.data.create_subject_splits --config configs/training.yaml
python -m src.evaluation.leakage_check --config configs/training.yaml
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml

python -m src.training.train_eegnet --config configs/training.yaml --fold 0
python -m src.training.train_eegnet --config configs/training.yaml --fold 1
python -m src.training.train_eegnet --config configs/training.yaml --fold 2
python -m src.training.train_eegnet --config configs/training.yaml --fold 3
python -m src.training.train_eegnet --config configs/training.yaml --fold 4

python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds --no-auto-train
python -m src.evaluation.test_inference_subjects --config configs/training.yaml --n-subjects 10
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-043 --no-auto-train
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-043 --all-folds
python -m app.gradio_app
```

Or run the fold 0 pipeline with the included PowerShell helper:

```powershell
.\run_local_fold0.ps1 -SubjectId sub-001
```

To reuse existing preprocessing or training outputs:

```powershell
.\run_local_fold0.ps1 -SubjectId sub-001 -SkipPreprocess
.\run_local_fold0.ps1 -SubjectId sub-001 -SkipPreprocess -SkipTrain
```

Open the local Gradio URL printed by the last command, usually:

```text
http://127.0.0.1:7860
```

For a public temporary Gradio link, use:

```powershell
$env:GRADIO_SHARE="1"
python -m app.gradio_app
```

## 3. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Modal or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Download Dataset

Recommended OpenNeuro command:

```bash
pip install -q --upgrade openneuro-py
openneuro-py download --dataset ds004504 --target-dir data/raw/openneuro_ds004504
```

Alternative GitHub source:

```bash
git clone https://github.com/OpenNeuroDatasets/ds004504.git data/raw/openneuro_ds004504
```

Expected Phase 1 groups:

- Healthy Control / `C`: 29 included, label `0`
- Alzheimer's Disease / `A`: 36 included, label `1`
- Frontotemporal Dementia / `F`: 23 excluded, label `-1`

The dataset size is suitable for Phase 1 proof-of-concept and baseline EEG representation learning. However, further validation with larger and external datasets is required before clinical use.

## 5. Prepare Labels

```bash
python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

Outputs:

- `data/labels/participants_processed.csv`
- `data/labels/ad_hc_subjects.csv`
- `data/labels/excluded_ftd_subjects.csv`
- `data/labels/label_mapping.json`

## 6. Create Leakage-Safe Subject Splits

```bash
python -m src.data.create_subject_splits --config configs/training.yaml
python -m src.evaluation.leakage_check --config configs/training.yaml
```

Outputs:

- `data/splits/subject_level_5fold.csv`
- `data/splits/split_summary.json`
- `outputs/reports/leakage_check.json`

Important rule: epochs from the same subject must never appear in both training and validation/test sets.

## 7. Preprocess EEG

Preprocess all labeled AD/HC subjects:

```bash
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all
```

Force-refresh stale preprocessing outputs after changing channels, epoch length, or filtering:

```bash
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
```

Preprocess one subject:

```bash
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --subject_id sub-001
```

Pipeline:

1. Channel selection and montage setup
2. Band-pass filtering `0.5-40 Hz`
3. Artifact/noise rejection using peak-to-peak amplitude threshold
4. Fixed-length epoch segmentation
5. Standardized model-ready EEG tensor

Outputs:

- `data/processed/epochs/*_epochs.npz`
- `data/processed/preprocessing_logs/*_preprocessing_log.json`
- `outputs/figures/*raw_eeg_plot.png`
- `outputs/figures/*filtered_eeg_plot.png`

## 8. Train EEGNet

The default `configs/training.yaml` is for real OpenNeuro training only and has:

```yaml
synthetic_fallback: false
max_train_epochs_per_subject: 60
max_eval_epochs_per_subject: null
```

Train all five subject-level folds before reporting final performance:

```powershell
python -m src.training.train_eegnet --config configs/training.yaml --fold 0
python -m src.training.train_eegnet --config configs/training.yaml --fold 1
python -m src.training.train_eegnet --config configs/training.yaml --fold 2
python -m src.training.train_eegnet --config configs/training.yaml --fold 3
python -m src.training.train_eegnet --config configs/training.yaml --fold 4
```

Outputs:

- `models/checkpoints/eegnet_fold{fold}_best.pth`
- `outputs/reports/fold{fold}_training_history.csv`
- `outputs/reports/fold{fold}_training_summary.json`
- `outputs/figures/fold{fold}_training_curve.png`

The model returns:

- Binary logits
- AD EEG Pattern probability
- `z_eeg`, shape `[256]`
- L2-normalized embedding

For local smoke testing without the dataset, use:

```bash
python -m src.training.train_eegnet --config configs/training_synthetic_test.yaml
```

## 9. Evaluate Subject-Level Metrics

Validate channel order and epoch tensor shapes:

```bash
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
```

Aggregate all five folds after training checkpoints are available:

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds --no-auto-train
```

Optional single-fold evaluation:

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --fold 0 --no-auto-train
```

Outputs:

- `outputs/reports/phase1_metrics.json`
- `outputs/reports/phase1_crossval_metrics.json`
- `outputs/reports/fold0_subject_predictions.csv`
- `outputs/figures/fold0_confusion_matrix.png`
- `outputs/figures/fold0_roc_curve.png`
- `outputs/figures/fold0_embedding_pca.png`
- `outputs/figures/fold0_embedding_tsne.png` when enough subjects are available

Use subject-level metrics as the primary result. Epoch-level scores can be useful for debugging, but they are not the final evaluation because many epochs come from the same subject.

Important metrics:

- AUC-ROC: best overall ranking metric for AD vs HC.
- F1-score: useful when AD/HC class counts are not perfectly balanced.
- Recall/sensitivity: important for detecting Alzheimer's EEG Pattern subjects.
- Precision: shows how reliable AD-positive predictions are.
- Accuracy: useful, but do not trust it alone.
- Confusion matrix: shows false positives and missed AD cases.
- Cross-validation mean/std: checks whether results are stable across folds.
- Embedding silhouette: optional check for whether `z_eeg` separates AD/HC embeddings.

The metric implementation is in `src/evaluation/metrics.py`. In the JSON output, sensitivity is reported as `recall`.

Local files to inspect after evaluation:

```text
outputs/reports/phase1_metrics.json
outputs/reports/phase1_crossval_metrics.json
outputs/reports/fold0_subject_predictions.csv
outputs/figures/fold0_confusion_matrix.png
outputs/figures/fold0_roc_curve.png
outputs/figures/fold0_embedding_pca.png
```

For final reporting, prefer all-fold results from `phase1_crossval_metrics.json` when all checkpoints are trained.

## 9.1 Test Inference Across Fold Models

After all five checkpoints and threshold calibration files exist, run a quick
10-subject inference sanity check across fold models `0-4`:

```powershell
python -m src.evaluation.test_inference_subjects --config configs/training.yaml --n-subjects 10
```

Outputs:

- `outputs/reports/inference_10_subjects_folds_0_4.csv`
- `outputs/reports/inference_10_subjects_folds_0_4.json`

The script selects 5 AD and 5 Healthy Control subjects from the generated label
file, runs each subject against all trained fold checkpoints, records whether
the subject was `train` or `val` for that fold, and reports overall/per-fold
accuracy, balanced accuracy, F1, precision, and recall.

## 9.2 Check Overfitting After Training

After each training run, inspect:

```text
outputs/figures/fold0_training_curve.png
outputs/reports/fold0_training_history.csv
```

The training curve includes:

- Train vs validation loss
- Train vs validation subject accuracy
- Train vs validation subject F1
- Train vs validation subject AUC

Common overfitting signs:

- Training loss keeps falling while validation loss rises.
- Training accuracy/F1/AUC becomes high while validation metrics stay flat or drop.
- A large train-validation metric gap appears after a few epochs.

If this happens, try stronger regularization, lower learning rate, fewer epochs, earlier stopping, more conservative preprocessing, or more subjects/folds before trusting the result.

## 10. Predict One Subject And Generate JSON

```bash
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-001
```

Outputs:

- `outputs/reports/sub-001_prediction_report.json`
- `outputs/figures/sub-001_probability_bar_chart.png`
- `outputs/figures/sub-001_epoch_probability_distribution.png`
- `models/embeddings/sub-001_z_eeg.npy`

The JSON includes:

- Dataset metadata
- Input metadata
- Preprocessing summary
- Subject-level prediction
- Confidence interpretation
- Epoch probability summary
- Probability bar chart data
- `z_eeg` shape and L2 norm
- Clinical disclaimer

## 11. Modal/Local Post-Training Smoke Test

After training on Modal, run this before presentation or submission:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --fold 0
```

For all trained folds:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --all-folds
```

This script does not start training. It checks:

- Label file and AD/HC labels
- Subject split file and no subject leakage
- Processed epoch files
- Fold checkpoint exists
- Subject-level evaluation runs with `auto_train=False`
- Prediction JSON report is generated
- `z_eeg_shape = [256]`, `l2_norm ~= 1.0`, and `availability_flag = 1`
- Clinical disclaimer is present
- Visual artifact files exist
- Gradio app can be built

It writes:

- `outputs/reports/modal_post_train_check.json`

For local synthetic smoke tests only, add:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-synth-hc-000 --fold 0 --allow-synthetic
```

Keep normal local unit tests separate:

```bash
pytest -q
```

## 12. Launch Gradio Demo

```bash
python -m app.gradio_app
```

The Gradio demo expects a trained checkpoint. It will show a clear error if
`models/checkpoints/eegnet_fold0_best.pth` is missing instead of starting a
training run inside the web app process.

If the demo was already running before code or config changes, stop it and start it again. Gradio launch settings and allowed file paths are only applied when the process starts.

The app can display visual files from local project outputs and Modal artifact paths. It also copies generated figures into:

```text
outputs/gradio_cache/<subject_id>/
```

The demo shows:

- Subject dropdown
- Raw EEG plot
- Filtered EEG plot
- Clean/rejected epoch summary
- Signal quality
- Prediction: Healthy Control / Alzheimer's EEG Pattern
- Confidence score
- Probability chart
- Epoch probability distribution
- `z_eeg` shape and L2 norm
- Clinical disclaimer
- Full JSON report

## 13. Code Architecture And Flow

This project is organized as small command-line modules. Each command uses the
same config files, so the local scripts, notebooks, and Gradio app all run the
same pipeline.

### 13.1 Configuration

Main files:

```text
configs/preprocessing.yaml
configs/training.yaml
configs/training_synthetic_test.yaml
```

`configs/training.yaml` inherits from `configs/preprocessing.yaml` through the
`_base_` key. The loader in `src/utils/config_loader.py` merges the base config
first, then applies training overrides.

Important config sections:

- `paths`: where raw data, processed epochs, reports, figures, checkpoints, and embeddings are stored.
- `preprocessing`: filter range, selected channels, ICA, epoch length, artifact rejection, and signal-quality thresholds.
- `data`: label file, split file, active fold, fold count, subject epoch limits, and synthetic fallback behavior.
- `model`: EEGNet channel filters, temporal kernel, dropout, and embedding dimension.
- `training`: batch size, learning rate, weight decay, early stopping, class weights, label smoothing, gradient clipping.
- `inference`: decision threshold, calibrated-threshold use, uncertainty margin, and risk cutoffs.

### 13.2 Label Preparation

Command:

```powershell
python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

Code path:

```text
src/data/prepare_labels.py
```

What it does:

1. Reads `data/raw/openneuro_ds004504/participants.tsv`.
2. Finds the diagnosis/group column.
3. Maps dataset groups:
   - `C` -> `0`, Healthy Control
   - `A` -> `1`, Alzheimer's EEG Pattern
   - `F` -> `-1`, Frontotemporal Dementia
4. Writes Phase 1 AD/HC labels.
5. Writes excluded FTD subjects separately.

Generated files:

```text
data/labels/participants_processed.csv
data/labels/ad_hc_subjects.csv
data/labels/excluded_ftd_subjects.csv
data/labels/label_mapping.json
```

### 13.3 Subject-Level Splits

Command:

```powershell
python -m src.data.create_subject_splits --config configs/training.yaml
python -m src.evaluation.leakage_check --config configs/training.yaml
```

Code paths:

```text
src/data/create_subject_splits.py
src/evaluation/leakage_check.py
```

What it does:

1. Loads `data/labels/ad_hc_subjects.csv`.
2. Uses `StratifiedGroupKFold`.
3. Groups by `subject_id`, not by epoch.
4. Writes train/validation assignments for folds `0-4`.
5. Checks that no subject appears in both train and validation inside the same fold.

This is critical because EEG epochs from the same subject are highly related.
Splitting by epoch would cause leakage and inflated accuracy.

Generated files:

```text
data/splits/subject_level_5fold.csv
data/splits/split_summary.json
outputs/reports/leakage_check.json
```

### 13.4 Preprocessing

Command:

```powershell
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
```

Main code path:

```text
src/preprocessing/preprocess_subject.py
```

Helper modules:

```text
src/data/load_bids.py
src/preprocessing/filter_eeg.py
src/preprocessing/epoching.py
src/preprocessing/artifact_rejection.py
src/preprocessing/time_frequency.py
```

What happens for each subject:

1. Find the subject EEG file, usually:

   ```text
   data/raw/openneuro_ds004504/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
   ```

2. Read raw EEG with MNE.
3. Enforce the selected 19-channel order:

   ```text
   Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
   F7, F8, T3, T4, T5, T6, Fz, Cz, Pz
   ```

4. Apply montage and channel aliases where needed.
5. Band-pass filter before resampling.
6. Resample to the configured target sampling rate.
7. Optionally run ICA artifact cleaning.
8. Segment into fixed-length overlapping epochs.
9. Reject noisy epochs using raw-scale peak-to-peak amplitude.
10. Standardize each epoch/channel for EEGNet input.
11. Save model-ready tensors.

Important output shape:

```text
epochs: [n_clean_epochs, 19, 1000]
```

With the current config, `1000` samples comes from:

```text
4 seconds x 250 Hz = 1000 samples
```

Generated files:

```text
data/processed/epochs/sub-XXX_epochs.npz
data/processed/preprocessing_logs/sub-XXX_preprocessing_log.json
outputs/figures/sub-XXX_raw_eeg_plot.png
outputs/figures/sub-XXX_filtered_eeg_plot.png
```

Poor-signal handling:

- A subject must meet `min_clean_epochs`.
- If preprocessing produces too few clean epochs, the subject is excluded from the Phase 1 training label file.
- Excluded poor-signal subjects are written to:

```text
data/labels/excluded_poor_signal_subjects.csv
```

### 13.5 Shape Validation

Command:

```powershell
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
```

Code path:

```text
src/evaluation/validate_preprocessed_shapes.py
```

What it checks:

- Every processed subject has at least `min_clean_epochs`.
- Every epoch tensor has the same channel/sample shape.
- Channel order matches `preprocessing.selected_channels`.
- No zero-epoch subject is silently used for training.

Output:

```text
outputs/reports/preprocessed_shape_validation.json
```

### 13.6 Model Code

Main model wrapper:

```text
src/models/eeg_encoder.py
```

Core encoder:

```text
src/models/eegnet.py
```

Projection head:

```text
src/models/projection_head.py
```

Forward pass:

```text
raw epoch tensor
-> EEGNetEncoder
-> ProjectionHead
-> 256D L2-normalized z_eeg
-> Linear classifier
-> AD/HC probabilities
```

The model returns:

```python
{
    "logits": logits,
    "probabilities": probabilities,
    "ad_probability": probabilities[:, 1],
    "z_eeg": z_eeg,
}
```

`z_eeg` is the representation vector used for PCA/t-SNE plots and embedding
exports.

### 13.7 Training

Command:

```powershell
python -m src.training.train_eegnet --config configs/training.yaml --fold 2
```

Code path:

```text
src/training/train_eegnet.py
```

Training flow:

1. Load config and active fold.
2. Ensure labels exist.
3. Ensure processed epochs exist.
4. Validate processed shapes.
5. Recreate subject splits if configured.
6. Load train and validation subjects for the fold.
7. Build `EpochDataset`.
8. Load all subject epochs into memory.
9. Build EEGNet model.
10. Use class-weighted cross-entropy.
11. Apply label smoothing.
12. Use AdamW optimizer with weight decay.
13. Clip gradients.
14. Track train/validation loss, F1, AUC, and subject accuracy.
15. Save the best checkpoint using validation subject AUC when available.
16. Stop early when validation performance does not improve.

Key anti-overfitting settings:

```yaml
dropout: 0.6
weight_decay: 0.001
label_smoothing: 0.05
grad_clip_norm: 1.0
early_stopping_patience: 7
```

Generated files per fold:

```text
models/checkpoints/eegnet_fold2_best.pth
outputs/reports/fold2_training_history.csv
outputs/reports/fold2_training_summary.json
outputs/figures/fold2_training_curve.png
```

### 13.8 Subject-Level Evaluation

Command:

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds --no-auto-train
```

Code path:

```text
src/evaluation/evaluate_subject_level.py
```

Metric helpers:

```text
src/evaluation/metrics.py
src/evaluation/plots.py
```

Evaluation flow:

1. Load the fold checkpoint.
2. Load only validation subjects for that fold.
3. Run all clean epochs through the model.
4. Average AD probabilities across epochs for each subject.
5. Average and L2-normalize epoch embeddings for subject-level `z_eeg`.
6. Compute subject-level metrics.
7. Find a validation-calibrated threshold.
8. Save confusion matrix, ROC curve, PCA, and t-SNE plots.

Main metrics:

- Accuracy
- Balanced accuracy
- F1
- Precision
- Recall/sensitivity
- Specificity
- AUC
- Confusion matrix
- Embedding silhouette

Generated files:

```text
outputs/reports/phase1_metrics.json
outputs/reports/phase1_crossval_metrics.json
outputs/reports/fold2_subject_predictions.csv
outputs/reports/fold2_threshold_calibration.json
outputs/figures/fold2_confusion_matrix.png
outputs/figures/fold2_roc_curve.png
outputs/figures/fold2_embedding_pca.png
outputs/figures/fold2_embedding_tsne.png
```

### 13.9 Inference

Command:

```powershell
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-043 --no-auto-train
```

Main code path:

```text
src/inference/predict_subject.py
```

Report builder:

```text
src/inference/report_generator.py
src/inference/output_schema.py
src/inference/generate_zeeg.py
```

Inference flow:

1. Read `data.active_fold` from `configs/training.yaml`.
2. Load the matching checkpoint:

   ```text
   models/checkpoints/eegnet_fold{active_fold}_best.pth
   ```

3. Preprocess the requested subject.
4. Run all clean epochs through the model.
5. Average epoch AD probabilities into one subject-level probability.
6. Load the calibrated threshold for the active fold if available.
7. Assign the subject-level prediction.
8. Mark the result uncertain if it is near the threshold.
9. Aggregate epoch embeddings into subject-level `z_eeg`.
10. Write JSON report and plots.

Generated files:

```text
outputs/reports/sub-043_prediction_report.json
outputs/figures/sub-043_probability_bar_chart.png
outputs/figures/sub-043_epoch_probability_distribution.png
models/embeddings/sub-043_z_eeg.npy
```

### 13.10 Ten-Subject Fold Inference Test

Command:

```powershell
python -m src.evaluation.test_inference_subjects --config configs/training.yaml --n-subjects 10
```

Code path:

```text
src/evaluation/test_inference_subjects.py
```

What it does:

1. Selects 5 AD and 5 Healthy Control subjects from the generated label file.
2. Loads checkpoints for folds `0-4`.
3. Loads each fold's calibrated threshold.
4. Runs every selected subject through every fold model.
5. Records whether that subject was `train` or `val` for each fold.
6. Computes overall, per-fold, and train-vs-validation metrics.

Generated files:

```text
outputs/reports/inference_10_subjects_folds_0_4.csv
outputs/reports/inference_10_subjects_folds_0_4.json
```

Use this as a quick inference sanity check. Final reported model performance
should still come from `phase1_crossval_metrics.json`.

### 13.11 Gradio App

Command:

```powershell
python -m app.gradio_app
```

App code:

```text
app/gradio_app.py
app/demo_utils.py
app/subject_options.py
```

How it works:

1. Builds a Gradio UI.
2. Loads demo subject options from the active fold.
3. Calls `src.inference.predict_subject`.
4. Displays raw/filtered EEG plots.
5. Displays probability charts and epoch probability distribution.
6. Displays the full prediction JSON.

The Gradio app uses:

```yaml
data:
  active_fold: 2
```

if that is what is currently set in `configs/training.yaml`. So with
`active_fold: 2`, the app loads:

```text
models/checkpoints/eegnet_fold2_best.pth
outputs/reports/fold2_threshold_calibration.json
```

### 13.12 End-To-End Data Flow

```text
OpenNeuro ds004504 raw EEG
-> prepare_labels
-> subject-level splits
-> preprocessing
-> processed epoch tensors
-> EEGNet training per fold
-> subject-level evaluation
-> calibrated thresholds
-> prediction report / Gradio demo
```

The central safety principle is:

```text
Split by subject, never by epoch.
```

## 14. Notebook Runner Style

Notebooks are intentionally thin. They call scripts rather than storing model code.

Example cells:

```python
!python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

```python
!python -m src.training.train_eegnet --config configs/training.yaml
```

```python
!python -m src.evaluation.evaluate_subject_level --config configs/training.yaml
```

```python
!python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-001
```

## 15. Modal Notebook Workflow

Recommended for Phase 1:

- Code path: `/root/eeg_ad_hc_phase1/`
- Dataset volume: `/mnt/eeg-dataset-vol/`
- Artifacts volume: `/mnt/eeg-artifacts-vol/`

Create volumes:

```bash
modal volume create eeg-dataset-vol
modal volume create eeg-artifacts-vol
```

Suggested Modal storage:

```text
/mnt/eeg-dataset-vol/
├── raw/openneuro_ds004504/
└── processed/

/mnt/eeg-artifacts-vol/
├── checkpoints/
├── embeddings/
├── outputs/
└── reports/
```

Update `configs/training.yaml` paths in Modal:

```yaml
paths:
  dataset_root: /mnt/eeg-dataset-vol/raw/openneuro_ds004504
  labels_dir: /mnt/eeg-artifacts-vol/labels
  splits_dir: /mnt/eeg-artifacts-vol/splits
  processed_epochs_dir: /mnt/eeg-dataset-vol/processed/epochs
  preprocessing_logs_dir: /mnt/eeg-dataset-vol/processed/preprocessing_logs
  checkpoints_dir: /mnt/eeg-artifacts-vol/checkpoints
  final_model_dir: /mnt/eeg-artifacts-vol/final
  embeddings_dir: /mnt/eeg-artifacts-vol/embeddings
  reports_dir: /mnt/eeg-artifacts-vol/reports
  figures_dir: /mnt/eeg-artifacts-vol/figures
  logs_dir: /mnt/eeg-artifacts-vol/logs
```

GPU recommendation:

- Use `L40S` if available.
- T4, L4, or A10 are acceptable lower-cost options.
- A100 is not required for Phase 1 EEGNet unless later CWT/Transformer experiments become large.

## 16. Tests

```bash
pytest
```

Tests verify:

- No subject leakage across folds
- EEGNet output shape
- `z_eeg` L2 normalization
- Required JSON report keys and safe wording
- Epoching and amplitude rejection helper behavior

## 17. Acceptance Criteria

Phase 1 is complete when one subject/sample can run through:

```text
Load EEG -> preprocess -> clean epochs -> train/infer EEGNet -> subject-level prediction -> z_eeg -> JSON report
```

Required evidence:

- FTD excluded
- Train/test split is subject-level
- No epoch leakage
- Metrics are subject-level
- `z_eeg_shape = [256]`
- `l2_norm ~= 1.0`
- `availability_flag = 1`
- Clinical disclaimer is present
