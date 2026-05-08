# Local Setup And Run Guide

This guide is only for running the project locally on Windows PowerShell.

## 1. Open Project Folder

```powershell
cd C:\Users\skspa\Downloads\cookie\eeg_ad_hc_phase1_fixed\eeg_ad_hc_phase1_fixed
```

## 2. Create And Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Download Dataset

```powershell
pip install --upgrade openneuro-py
openneuro-py download --dataset ds004504 --target-dir data/raw/openneuro_ds004504
```

Expected raw folder:

```text
data/raw/openneuro_ds004504
```

## 5. Prepare Labels

```powershell
python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

Check:

```text
data/labels/ad_hc_subjects.csv
data/labels/excluded_ftd_subjects.csv
```

## 6. Preprocess All Subjects

```powershell
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
```

This can take time because ICA is enabled.

Check:

```text
data/processed/epochs/
data/processed/preprocessing_logs/
outputs/figures/
```

## 7. Create Splits And Check Leakage

```powershell
python -m src.data.create_subject_splits --config configs/training.yaml
python -m src.evaluation.leakage_check --config configs/training.yaml
```

Check:

```text
data/splits/subject_level_5fold.csv
outputs/reports/leakage_check.json
```

## 8. Validate Preprocessed Shapes

```powershell
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
```

Expected:

```text
Preprocessed shape validation: PASS
```

If it fails, inspect:

```text
outputs/reports/preprocessed_shape_validation.json
```

## 9. Train All Folds

```powershell
python -m src.training.train_eegnet --config configs/training.yaml --fold 0
python -m src.training.train_eegnet --config configs/training.yaml --fold 1
python -m src.training.train_eegnet --config configs/training.yaml --fold 2
python -m src.training.train_eegnet --config configs/training.yaml --fold 3
python -m src.training.train_eegnet --config configs/training.yaml --fold 4
```

Check:

```text
models/checkpoints/eegnet_fold0_best.pth
models/checkpoints/eegnet_fold1_best.pth
models/checkpoints/eegnet_fold2_best.pth
models/checkpoints/eegnet_fold3_best.pth
models/checkpoints/eegnet_fold4_best.pth
```

## 10. Evaluate All Folds

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds --no-auto-train
```

Check:

```text
outputs/reports/phase1_crossval_metrics.json
outputs/reports/fold0_threshold_calibration.json
outputs/reports/fold1_threshold_calibration.json
outputs/reports/fold2_threshold_calibration.json
outputs/reports/fold3_threshold_calibration.json
outputs/reports/fold4_threshold_calibration.json
```

## 11. Test 10 Subjects Across Fold Models

```powershell
python -m src.evaluation.test_inference_subjects --config configs/training.yaml --n-subjects 10
```

Check:

```text
outputs/reports/inference_10_subjects_folds_0_4.csv
outputs/reports/inference_10_subjects_folds_0_4.json
```

## 12. Select Best Fold For Demo

Set the best fold in:

```text
configs/training.yaml
```

Example:

```yaml
data:
  active_fold: 2
```

With this setting, inference and Gradio use:

```text
models/checkpoints/eegnet_fold2_best.pth
outputs/reports/fold2_threshold_calibration.json
```

## 13. Run One Prediction

```powershell
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-043 --no-auto-train
```

Check:

```text
outputs/reports/sub-043_prediction_report.json
```

## 14. Build Training Dashboard HTML

```powershell
python -m scripts.build_training_dashboard --config configs/training.yaml
```

Open:

```text
outputs/reports/training_dashboard.html
```

## 15. Launch Gradio

```powershell
python -m app.gradio_app
```

Open the printed URL, usually:

```text
http://127.0.0.1:7860
```

## 16. Local Troubleshooting

If `python` is not recognized:

- activate the venv again
- or run with full path:

```powershell
.\venv\Scripts\python.exe -m app.gradio_app
```

If validation fails:

```powershell
Get-Content outputs/reports/preprocessed_shape_validation.json
```

If Gradio uses the wrong fold:

```text
Check configs/training.yaml -> data.active_fold
```

If inference says calibration missing:

```powershell
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --fold 2 --no-auto-train
```

