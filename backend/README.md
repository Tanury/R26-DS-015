# EEG AD/HC Phase 1

Leakage-safe Phase 1 EEG representation learning prototype for OpenNeuro `ds004504`.

The project focuses on:

- AD vs Healthy Control binary classification.
- EEG preprocessing with configurable channels, `0.5-40 Hz` filtering, epoching, and amplitude rejection.
- EEGNet baseline model.
- Subject-level evaluation using `StratifiedGroupKFold`.
- 256D L2-normalized `z_eeg` embeddings.
- Safe research JSON reports and Gradio demo.

`configs/training.yaml` is real-data only: `synthetic_fallback: false`.
Use `configs/training_synthetic_test.yaml` only for local smoke tests.
Training uses `max_train_epochs_per_subject`; evaluation and inference use all clean epochs by default.

For local Windows execution, start in this folder, create a virtual environment, install requirements, download `ds004504` into `data/raw/openneuro_ds004504`, then run the local command sequence in [guide.md](guide.md).

After local or Modal training, run:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --fold 0
```

This verifies the checkpoint, leakage-safe split, subject-level evaluation, prediction JSON, `z_eeg`, visual artifacts, and Gradio app build without starting a new training run.

See [guide.md](guide.md) for local setup, metrics/overfitting checks, Modal workflow, and run commands.
