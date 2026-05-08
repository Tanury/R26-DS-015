# Folder Structure And Code Explanation

This document explains the main folders and important files in the project.

## Root Files

```text
README.md
guide.md
setup.md
requirements.txt
run_local_fold0.ps1
.gitignore
```

### `README.md`

Short project overview. Use it as the first page for GitHub.

### `guide.md`

Full project guide with commands, architecture, and workflow.

### `setup.md`

Setup notes, especially for Modal or notebook-style execution.

### `requirements.txt`

Python dependencies.

### `run_local_fold0.ps1`

PowerShell helper for a local fold-0 smoke run.

### `.gitignore`

Prevents raw EEG, processed data, model checkpoints, outputs, caches, and local venv files from being pushed.

## `configs/`

Configuration folder.

```text
configs/preprocessing.yaml
configs/training.yaml
configs/training_synthetic_test.yaml
```

### `configs/preprocessing.yaml`

Controls EEG preprocessing:

- raw dataset path
- selected channels
- filter range
- resampling rate
- epoch length
- ICA
- artifact rejection
- output paths

### `configs/training.yaml`

Main real-data training config. It inherits preprocessing settings and adds:

- fold count
- active fold
- label/split paths
- checkpoint paths
- model hyperparameters
- optimizer settings
- early stopping
- inference threshold behavior

### `configs/training_synthetic_test.yaml`

Small synthetic config for quick smoke tests only. Do not report scientific results from it.

## `data/`

Data folder. Most of this is generated or downloaded and should not be committed.

```text
data/raw/
data/labels/
data/splits/
data/processed/
```

### `data/raw/`

Downloaded OpenNeuro dataset:

```text
data/raw/openneuro_ds004504/
```

This contains BIDS-style subject folders such as:

```text
sub-001/
sub-002/
...
```

### `data/labels/`

Generated labels:

```text
ad_hc_subjects.csv
excluded_ftd_subjects.csv
excluded_poor_signal_subjects.csv
label_mapping.json
participants_processed.csv
```

### `data/splits/`

Subject-level train/validation fold files:

```text
subject_level_5fold.csv
split_summary.json
```

### `data/processed/`

Preprocessed EEG tensors and logs:

```text
data/processed/epochs/
data/processed/preprocessing_logs/
```

## `src/`

Main source code.

```text
src/data/
src/preprocessing/
src/models/
src/training/
src/evaluation/
src/inference/
src/utils/
```

## `src/data/`

Data loading, labels, and splits.

### `prepare_labels.py`

Reads OpenNeuro `participants.tsv`, maps group codes to labels, excludes FTD, and writes label CSV files.

### `create_subject_splits.py`

Creates 5 subject-level folds using `StratifiedGroupKFold`.

### `dataset.py`

Defines `EpochDataset`. It loads preprocessed epoch tensors and returns:

```python
x, y, subject_id
```

where:

- `x`: epoch tensor
- `y`: label
- `subject_id`: subject group key

### `load_bids.py`

Finds and reads subject EEG files from the OpenNeuro BIDS-style folder.

## `src/preprocessing/`

EEG preprocessing code.

### `preprocess_subject.py`

Main preprocessing entrypoint. It handles:

- single-subject preprocessing
- all-subject preprocessing
- force-refreshing outputs
- channel selection
- ICA summary
- artifact rejection summary
- poor-signal exclusion
- visual plots

### `filter_eeg.py`

Applies band-pass filtering and resampling.

### `epoching.py`

Creates fixed-length overlapping EEG epochs.

### `artifact_rejection.py`

Contains ICA helper logic and amplitude-based epoch rejection.

### `time_frequency.py`

Contains raw tensor and optional STFT/CWT feature helpers.

## `src/models/`

Neural network model code.

### `eegnet.py`

EEGNet encoder. It extracts EEG features from `[batch, 1, channels, samples]`.

### `projection_head.py`

Projects encoder features into a 256-dimensional L2-normalized representation.

### `eeg_encoder.py`

Combines EEGNet encoder, projection head, and classifier. Returns logits, probabilities, AD probability, and `z_eeg`.

## `src/training/`

Training code.

### `train_eegnet.py`

Main training script. It:

- loads config
- checks labels/splits/epochs
- builds datasets
- trains EEGNet
- saves best checkpoint
- writes training history
- plots training curves

### `callbacks.py`

Early stopping helper.

### `losses.py`

Class-weight helper for imbalanced AD/HC counts.

## `src/evaluation/`

Evaluation and reporting helpers.

### `evaluate_subject_level.py`

Runs subject-level validation. It aggregates epoch predictions per subject and writes metrics/plots.

### `metrics.py`

Computes:

- accuracy
- balanced accuracy
- F1
- precision
- recall
- specificity
- AUC
- confusion matrix
- threshold calibration
- embedding silhouette

### `plots.py`

Creates confusion matrix, ROC, PCA, t-SNE, probability, and epoch-distribution plots.

### `leakage_check.py`

Checks that no subject appears in both train and validation within a fold.

### `validate_preprocessed_shapes.py`

Checks processed epoch tensors and channel order.

### `test_inference_subjects.py`

Runs 10 selected subjects across fold checkpoints `0-4`.

### `modal_post_train_check.py`

Smoke test for labels, splits, checkpoint, evaluation, inference report, and Gradio build.

## `src/inference/`

Single-subject inference and report generation.

### `predict_subject.py`

Loads the active fold checkpoint, preprocesses the subject, runs inference, aggregates epoch probabilities, and saves the JSON report.

### `report_generator.py`

Builds the final prediction report structure.

### `output_schema.py`

Contains risk-level and disclaimer helpers.

### `generate_zeeg.py`

Aggregates per-epoch embeddings into one subject-level `z_eeg`.

## `src/utils/`

Shared utilities.

### `config_loader.py`

Loads YAML configs and resolves relative paths.

### `file_utils.py`

Writes JSON and handles file helpers.

### `logger.py`

Configures logging.

### `seed.py`

Sets random seeds and stable subject hashing.

## `app/`

Gradio demo.

### `gradio_app.py`

Builds and launches the UI.

### `demo_utils.py`

Calls inference and prepares visual outputs for Gradio.

### `subject_options.py`

Builds subject dropdown options from labels/splits.

## `outputs/`

Generated reports and figures.

```text
outputs/reports/
outputs/figures/
outputs/logs/
outputs/gradio_cache/
```

Do not commit this folder unless intentionally sharing example artifacts.

## `models/`

Generated model artifacts.

```text
models/checkpoints/
models/embeddings/
models/final/
```

Do not commit checkpoints to normal Git history.

## `notebooks/`

Notebook experiments. Notebooks should call scripts instead of duplicating model code.

## `tests/`

Unit tests for config, metrics, preprocessing, model output, subject splitting, and report schema.

