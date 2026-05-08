# Important Notes, Assumptions, And Decisions

This document explains the most important assumptions, design decisions, and interpretation rules.

## 1. Why This Is Research-Only

The model predicts an `Alzheimer's EEG Pattern`, not a clinical diagnosis.

Reasons:

- The dataset is small.
- The task is binary AD vs Healthy Control only.
- FTD is excluded.
- There is no external validation dataset yet.
- EEG alone is not enough for clinical diagnosis.

Use wording like:

```text
Research and decision-support demonstration only.
Not a clinical diagnosis.
```

## 2. Dataset Assumptions

The current pipeline assumes:

- OpenNeuro `ds004504`.
- Resting-state closed-eyes EEG.
- 19 selected scalp EEG channels.
- AD group code `A`.
- Healthy Control group code `C`.
- FTD group code `F`, excluded from Phase 1.

If a new EEG dataset has different task, montage, diagnosis labels, sampling rate, or channel naming, accuracy may not transfer.

## 3. Why Subject-Level Splitting Is Required

EEG data is split into many epochs/windows. Epochs from the same subject are very similar.

If epochs from one subject appear in both train and validation, the model can memorize subject-specific signal and produce fake high accuracy.

Therefore, this project uses:

```text
StratifiedGroupKFold
group_key = subject_id
```

This ensures:

```text
one subject belongs to train OR validation for a fold, never both
```

## 4. What Is A Fold?

A fold is one train/validation split.

With `n_splits: 5`, the dataset is divided into five validation groups. Each fold trains on about 80% of subjects and validates on about 20%.

Example:

```text
Fold 0: train subjects A, validate subjects B
Fold 1: train subjects B/C/D/E, validate another group
...
Fold 4: final validation group
```

Every subject is used as validation once across the five folds.

## 5. Why Folds 0-4 Are Trained

Training only fold 0 can be misleading. It may be an easy or hard validation split.

Training folds `0-4` gives:

- more stable performance estimate
- mean accuracy
- standard deviation
- better view of generalization
- evidence that results are not fold-specific luck

Final reporting should use:

```text
outputs/reports/phase1_crossval_metrics.json
```

not only one fold.

## 6. How The Best Fold Model Is Selected

Each fold saves its own best checkpoint:

```text
models/checkpoints/eegnet_fold0_best.pth
models/checkpoints/eegnet_fold1_best.pth
models/checkpoints/eegnet_fold2_best.pth
models/checkpoints/eegnet_fold3_best.pth
models/checkpoints/eegnet_fold4_best.pth
```

The best single fold model is selected by looking at:

1. Validation AUC.
2. Validation accuracy.
3. Balanced accuracy.
4. F1.
5. Embedding silhouette.
6. Training curve stability.
7. Whether performance is too perfect too early.

In the current run, fold 2 is the best single model because it has strong validation metrics and embedding separation.

Use fold 2 for demo/inference by setting:

```yaml
data:
  active_fold: 2
```

in:

```text
configs/training.yaml
```

## 7. How Gradio Selects The Model

Gradio calls:

```text
src.inference.predict_subject
```

Inference reads:

```yaml
data:
  active_fold: 2
```

from:

```text
configs/training.yaml
```

So if `active_fold: 2`, Gradio uses:

```text
models/checkpoints/eegnet_fold2_best.pth
outputs/reports/fold2_threshold_calibration.json
```

If you want Gradio to use another fold, change `active_fold`.

## 8. Accuracy During Training

Training produces two kinds of accuracy:

1. Epoch-level metrics.
2. Subject-level metrics.

Subject-level metrics are more important because the final prediction is one result per subject.

Epoch-level metrics can help debug model learning, but they can be misleading because one subject contributes many epochs.

## 9. Accuracy After Training

After training, run:

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds --no-auto-train
```

Use:

```text
outputs/reports/phase1_crossval_metrics.json
```

Important values:

- mean accuracy
- mean balanced accuracy
- mean AUC
- mean F1
- recall/sensitivity
- specificity
- standard deviation

Do not report only the best fold as final accuracy. The best fold is useful for demo, not for final model performance.

## 10. Accuracy Of Inference Response

A single prediction report is not an accuracy measurement. It is one subject's predicted class and probability.

Example fields:

```text
prediction
ad_eeg_pattern_probability
decision_threshold
is_uncertain
margin_from_threshold
clean_epochs_used
signal_quality
embedding_consistency
```

Trust is higher when:

- signal quality is Good
- many clean epochs are used
- probability is far from threshold
- `is_uncertain` is false
- embedding consistency is high
- the subject is validation/test, not training

Trust is lower when:

- probability is near threshold
- few clean epochs are available
- signal quality is Poor
- epoch probability standard deviation is high
- the subject is from a different dataset/task/montage

## 11. How Overfitting Is Handled

Current overfitting controls:

- subject-level splitting
- no epoch leakage
- dropout
- AdamW weight decay
- label smoothing
- gradient clipping
- early stopping
- learning-rate scheduling
- validation checkpoint selection
- all-fold cross-validation
- minimum clean-epoch gate

Training is suspicious when:

- train AUC reaches 1.0 but validation AUC stays low
- training loss decreases while validation loss rises
- validation accuracy stops improving but training accuracy keeps rising
- one fold is much better than all others

## 12. Threshold Calibration

Default classification threshold is usually `0.5`.

This project also computes a validation-calibrated threshold per fold:

```text
outputs/reports/fold2_threshold_calibration.json
```

The calibrated threshold can improve balanced accuracy, especially if probabilities are not perfectly calibrated.

Important: report the threshold whenever showing predictions.

## 13. Important Questions To Answer In Reports

When presenting results, answer:

1. How many subjects were used?
2. Was FTD excluded?
3. Was splitting subject-level?
4. Was leakage checked?
5. How many folds were trained?
6. What is mean AUC across folds?
7. What is mean balanced accuracy across folds?
8. What is the best fold and why?
9. Which fold does Gradio use?
10. Are predictions calibrated?
11. What threshold was used?
12. Are uncertain predictions clearly marked?
13. Were poor-signal subjects excluded?
14. Is there external validation?
15. Is the disclaimer present?

## 14. Recommended Final Wording

Use:

```text
The model achieved subject-level cross-validation performance on OpenNeuro ds004504 for AD-vs-HC research classification. Predictions are EEG-pattern outputs and are not clinical diagnoses.
```

Avoid:

```text
The model diagnoses Alzheimer's disease.
```

