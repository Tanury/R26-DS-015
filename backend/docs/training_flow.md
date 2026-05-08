# Training Flow

This document explains how training is performed in this EEG AD/HC Phase 1 project.

The project trains an EEGNet baseline on OpenNeuro `ds004504` for binary subject-level classification:

- `0`: Healthy Control
- `1`: Alzheimer's EEG Pattern

The model is a research/demo model. It is not a clinical diagnostic system.

## 1. Full Training Pipeline

Recommended command order:

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
```

## 2. Data Preparation

Labels are created from:

```text
data/raw/openneuro_ds004504/participants.tsv
```

The label preparation code is:

```text
src/data/prepare_labels.py
```

It maps:

```text
C -> Healthy Control -> label 0
A -> Alzheimer's EEG Pattern -> label 1
F -> Frontotemporal Dementia -> excluded from Phase 1
```

Generated files:

```text
data/labels/ad_hc_subjects.csv
data/labels/excluded_ftd_subjects.csv
data/labels/label_mapping.json
```

## 3. Preprocessing

The preprocessing entrypoint is:

```text
src/preprocessing/preprocess_subject.py
```

Main preprocessing actions:

1. Read raw EEG.
2. Select the configured 19 EEG channels.
3. Apply standard montage.
4. Band-pass filter `0.5-40 Hz`.
5. Resample to `250 Hz`.
6. Optionally run ICA artifact cleaning.
7. Segment EEG into fixed-length epochs.
8. Reject high-amplitude artifact epochs.
9. Standardize each epoch/channel.
10. Save model-ready tensors.

Expected tensor shape:

```text
[n_clean_epochs, 19, 1000]
```

The `1000` sample length comes from:

```text
4 seconds x 250 Hz = 1000
```

Generated files:

```text
data/processed/epochs/sub-XXX_epochs.npz
data/processed/preprocessing_logs/sub-XXX_preprocessing_log.json
outputs/figures/sub-XXX_raw_eeg_plot.png
outputs/figures/sub-XXX_filtered_eeg_plot.png
```

## 4. Shape Validation

Validation command:

```powershell
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
```

Code:

```text
src/evaluation/validate_preprocessed_shapes.py
```

It checks:

- every subject has enough clean epochs
- channel order is correct
- tensor shapes match
- zero-epoch subjects are not silently used

Output:

```text
outputs/reports/preprocessed_shape_validation.json
```

## 5. Fold Splits

The split code is:

```text
src/data/create_subject_splits.py
```

It uses `StratifiedGroupKFold`.

The important part is the `group_key`:

```text
subject_id
```

That means epochs from one subject never appear in both train and validation for the same fold.

Output:

```text
data/splits/subject_level_5fold.csv
```

Leakage check:

```powershell
python -m src.evaluation.leakage_check --config configs/training.yaml
```

Output:

```text
outputs/reports/leakage_check.json
```

## 6. Model Training

Training code:

```text
src/training/train_eegnet.py
```

Model wrapper:

```text
src/models/eeg_encoder.py
```

Core EEGNet:

```text
src/models/eegnet.py
```

Projection head:

```text
src/models/projection_head.py
```

Training command for one fold:

```powershell
python -m src.training.train_eegnet --config configs/training.yaml --fold 2
```

Training flow:

1. Load config.
2. Load fold train/validation subjects.
3. Load preprocessed epochs into `EpochDataset`.
4. Build EEGNet model.
5. Train with class-weighted cross entropy.
6. Apply label smoothing.
7. Apply AdamW weight decay.
8. Clip gradients.
9. Track train/validation metrics.
10. Save the best checkpoint by validation subject AUC.
11. Stop early when validation performance stops improving.

Generated per fold:

```text
models/checkpoints/eegnet_fold2_best.pth
outputs/reports/fold2_training_history.csv
outputs/reports/fold2_training_summary.json
outputs/figures/fold2_training_curve.png
```

## 7. Evaluation

Evaluation code:

```text
src/evaluation/evaluate_subject_level.py
```

Metrics code:

```text
src/evaluation/metrics.py
```

Plot code:

```text
src/evaluation/plots.py
```

All-fold evaluation command:

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds --no-auto-train
```

Evaluation flow:

1. Load each fold checkpoint.
2. Load that fold's validation subjects.
3. Predict each clean epoch.
4. Average epoch probabilities into subject probability.
5. Compute subject-level metrics.
6. Calibrate threshold per fold.
7. Save plots and reports.

Important outputs:

```text
outputs/reports/phase1_crossval_metrics.json
outputs/reports/fold2_subject_predictions.csv
outputs/reports/fold2_threshold_calibration.json
outputs/figures/fold2_confusion_matrix.png
outputs/figures/fold2_roc_curve.png
outputs/figures/fold2_embedding_pca.png
outputs/figures/fold2_embedding_tsne.png
```

## 8. Inference

Inference code:

```text
src/inference/predict_subject.py
```

It uses:

```yaml
data:
  active_fold: 2
```

from:

```text
configs/training.yaml
```

If `active_fold: 2`, inference loads:

```text
models/checkpoints/eegnet_fold2_best.pth
outputs/reports/fold2_threshold_calibration.json
```

Inference command:

```powershell
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-043 --no-auto-train
```

Generated files:

```text
outputs/reports/sub-043_prediction_report.json
outputs/figures/sub-043_probability_bar_chart.png
outputs/figures/sub-043_epoch_probability_distribution.png
models/embeddings/sub-043_z_eeg.npy
```

## 9. Ten-Subject Inference Test

Code:

```text
src/evaluation/test_inference_subjects.py
```

Command:

```powershell
python -m src.evaluation.test_inference_subjects --config configs/training.yaml --n-subjects 10
```

It tests 5 AD and 5 Healthy Control subjects across fold models `0-4`.

Outputs:

```text
outputs/reports/inference_10_subjects_folds_0_4.csv
outputs/reports/inference_10_subjects_folds_0_4.json
```

Use this as an inference sanity check, not as the only final metric.

