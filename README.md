# Speech-Based Dementia Risk Screening

Phase 1 research prototype for dementia risk screening from speech/audio features using the DementiaBank Pitt Cookie Theft task.

The current implementation supports:

- audio feature extraction from speech files
- audio-only model training through notebooks
- final evaluation and explainability reports
- FastAPI inference for uploaded audio
- Gradio dashboard inference after training

Clinical note: this project is for research-based dementia risk screening only. It is not a clinical diagnosis.

## Project Structure

```text
app/
  main.py                         FastAPI backend
  config.py                       model and path settings
  services/
    audio_features.py             audio feature extraction
    audio_predictor.py            model loading and prediction formatting
  gradio_dashboard.py             Gradio inference dashboard

notebooks/
  01_parse_cha_files.ipynb
  02_train_transcript_model .ipynb
  03_extract_audio_features .ipynb
  04_train_audio_model .ipynb
  05_evaluation_explainability.ipynb
  models/
    audio_dementia_model.joblib
    audio_feature_columns.json
    text_dementia_model.joblib
  reports/
    final_evaluation_summary.json
    audio_test_predictions.csv
    ...

data/
  raw/                            local DementiaBank files, not committed
  processed/
  splits/

scripts/
  test_audio_prediction.py        local/API prediction test helper

run_gradio_dashboard.cmd          Windows launcher for Gradio
requirements.txt
```

## Setup

Use Python 3.10 or newer. Python 3.12 is also supported by the current environment.

### Windows

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Placement

Place DementiaBank Pitt Cookie Theft files under:

```text
data/raw/Pitt/Control/cookie/
data/raw/Pitt/Dementia/cookie/
```

Expected examples:

```text
data/raw/Pitt/Control/cookie/021-4.mp3
data/raw/Pitt/Control/cookie/021-4.cha
data/raw/Pitt/Dementia/cookie/035-1.mp3
data/raw/Pitt/Dementia/cookie/035-1.cha
```

Raw data is ignored by Git because it is large and may require dataset access permission.

## Training Workflow

Run the notebooks in this order.

### 1. Parse CHAT transcripts

Open and run:

```text
notebooks/01_parse_cha_files.ipynb
```

Main output:

```text
data/processed/transcripts_dataset.csv
notebooks/data/processed/transcripts_dataset.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

### 2. Train transcript model

Open and run:

```text
notebooks/02_train_transcript_model .ipynb
```

Main output:

```text
notebooks/models/text_dementia_model.joblib
notebooks/reports/text_model_metadata.json
notebooks/reports/text_test_predictions.csv
```

### 3. Extract audio features

Open and run:

```text
notebooks/03_extract_audio_features .ipynb
```

Main output:

```text
notebooks/data/processed/audio_features_dataset.csv
notebooks/data/splits/audio_train.csv
notebooks/data/splits/audio_val.csv
notebooks/data/splits/audio_test.csv
notebooks/reports/audio_feature_extraction_metadata.json
```

### 4. Train audio model

Open and run:

```text
notebooks/04_train_audio_model .ipynb
```

Main output required by the Gradio dashboard:

```text
notebooks/models/audio_dementia_model.joblib
notebooks/models/audio_feature_columns.json
notebooks/reports/audio_model_metadata.json
notebooks/reports/audio_test_predictions.csv
```

This notebook also contains the dashboard-style inference function logic. The deployed Gradio app uses the same prediction path through:

```text
app/services/audio_predictor.py
```

### 5. Final evaluation and explainability

Open and run:

```text
notebooks/05_evaluation_explainability.ipynb
```

Main output used by the dashboard:

```text
notebooks/reports/final_evaluation_summary.json
notebooks/reports/final_evaluation_explainability_report.txt
notebooks/reports/final_model_comparison.csv
notebooks/reports/final_sample_prediction_outputs.json
```

## Run Inference After Training

After the audio model and feature column files are generated, run inference using one of these options.

## Option A: Gradio Dashboard

### Windows

```bat
run_gradio_dashboard.cmd
```

Then open:

```text
http://127.0.0.1:7860
```

### Cross-platform command

```bash
python -m app.gradio_dashboard
```

The dashboard supports three inference modes:

- Audio feature row: select a test row such as `035-1 | Dementia`
- Upload audio: upload `.mp3`, `.wav`, `.m4a`, or `.flac`
- Manual features: paste JSON feature values

Example no-leakage output for uploaded control file `006-4.mp3`:

```json
{
  "input_metadata": {
    "file_id": "006-4.mp3",
    "input_type": "uploaded audio file"
  },
  "prediction": {
    "predicted_class": "Control",
    "risk_level": "Medium",
    "probabilities": {
      "Control": 55.58,
      "Dementia": 44.42
    }
  },
  "clinical_note": "Research-based dementia risk screening only. Not a clinical diagnosis."
}
```

Note: earlier perfect audio metrics were caused by accidental target leakage from `label_id.1`.
The current training configuration removes label and metadata columns before fitting.

## Option B: FastAPI Backend

Start the API:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/v1/health
```

Audio prediction endpoint:

```text
POST http://127.0.0.1:8000/api/v1/predict/audio-file
```

## Option C: Local Script Test

Run prediction directly without the API:

```bash
python scripts/test_audio_prediction.py "data/raw/Pitt/Dementia/cookie/035-1.mp3" --mode local
```

Run prediction through FastAPI:

```bash
python scripts/test_audio_prediction.py "data/raw/Pitt/Dementia/cookie/035-1.mp3" --mode api
```

## Important Generated Files

The Gradio dashboard expects these files to exist:

```text
notebooks/models/audio_dementia_model.joblib
notebooks/models/audio_feature_columns.json
notebooks/data/splits/audio_test.csv
notebooks/reports/audio_model_metadata.json
notebooks/reports/final_evaluation_summary.json
```

If the dashboard fails to load model predictions, rerun:

```text
03_extract_audio_features .ipynb
04_train_audio_model .ipynb
05_evaluation_explainability.ipynb
```

## Phase 1 Scope

Phase 1 is the speech/audio biomarker component:

- extract acoustic features from audio
- train the audio-only dementia classifier
- evaluate model accuracy, dementia recall, F1, and AUC
- show prediction, actual label, accuracy label, risk level, probabilities, and feature indicators in Gradio

Later project phases can extend this into multimodal fusion with retinal, MRI, EEG, and biomarker inputs.
