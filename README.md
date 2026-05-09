# BIO_BACKEND - Neurological Risk Assessment

FastAPI and Gradio application for Phase 1 neurological risk inference using clinical and
fluid biomarker inputs. The system predicts:

- Disease pattern: `AD`, `PD`, `MS`, or `Control`
- Neurological risk: `Low`, `Medium`, or `High`
- Confidence/accuracy label
- Class probabilities
- SHAP-based important features
- Clinical conclusion and recommendation

This is a clinical decision-support prototype, not a final medical diagnosis.

## Project Structure

```text
bio_backend/
  app/
    config.py
    main.py
    schemas.py
    gradio_dashboard.py
    models/
      disease_model.pkl
      disease_label_encoder.pkl
      disease_cat_encoders.pkl
      disease_num_imputer.pkl
      disease_shap_importance.csv
      risk_model.pkl
      risk_label_mapping.pkl
      risk_cat_encoders.pkl
      risk_num_imputer.pkl
      risk_shap_importance.csv
    services/
      conclusion_service.py
      feature_service.py
      prediction_service.py
  run.py
  run_dashboard.py
  test_backend.py
  requirements.txt
  README.md
```

## Requirements

- Python 3.10 or newer
- Saved model artifacts inside `app/models/`
- Recommended: a virtual environment

Required Python packages are listed in [requirements.txt](requirements.txt).

## Full Setup

### 1. Open the project folder

```bash
cd "F:\research aarbi\bio_backend"
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check model files

The app expects these files in `app/models/`:

```text
disease_model.pkl
disease_label_encoder.pkl
disease_num_imputer.pkl
disease_cat_encoders.pkl
disease_shap_importance.csv
risk_model.pkl
risk_label_mapping.pkl
risk_num_imputer.pkl
risk_cat_encoders.pkl
risk_shap_importance.csv
```

If these files are missing, copy them from the trained model output folder into `app/models/`.

### 5. Run the backend test

```bash
python test_backend.py
```

This checks:

- Required model files
- Pickle/model loading
- SHAP CSV files
- Direct prediction service
- FastAPI `/predict` endpoint if the API server is already running

If only the API connection test fails, start the FastAPI server and run the test again.

## Run FastAPI

Start the API:

```bash
python run.py
```

Alternative:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | API status message |
| GET | `/health` | Model load status |
| POST | `/predict` | Disease and risk prediction |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

## Example API Request

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":72,\"sex\":\"Male\",\"moca_total_score\":22,\"updrs_part_i\":14,\"updrs_part_ii\":26,\"updrs_part_iii\":58,\"updrs_part_iv\":9,\"disease_duration_years\":8.0,\"amyloid_beta_42_pg_ml\":690,\"p_tau181_pg_ml\":1.4,\"t_tau_pg_ml\":180,\"nfl_pg_ml\":29.0,\"gfap_pg_ml\":130,\"alpha_synuclein_pg_ml\":1350}"
```

## Run Gradio Dashboard

Start the Gradio UI:

```bash
python run_dashboard.py
```

Open:

```text
http://127.0.0.1:7860
```

To use another port:

Windows PowerShell:

```powershell
$env:GRADIO_PORT="7861"
python run_dashboard.py
```

macOS/Linux:

```bash
GRADIO_PORT=7861 python run_dashboard.py
```

## Gradio Workflow

The dashboard has two input tabs:

- `Dashboard Input`: editable form fields for the Phase 1 clinical and biomarker profile.
- `JSON Input`: paste a JSON object and click `Apply JSON to dashboard` to populate the form.

After applying JSON or editing the form manually, click `Run inference`.

The UI shows:

- Predicted neurological risk label
- Predicted disease pattern
- Disease confidence
- Risk confidence
- Accuracy label
- Disease probabilities
- Risk probabilities
- Important features
- Clinical interpretation
- Recommendation

## JSON Input Example

Paste this into the `JSON Input` tab:

```json
{
  "age": 72,
  "sex": "Male",
  "moca_total_score": 22,
  "updrs_part_i": 14,
  "updrs_part_ii": 26,
  "updrs_part_iii": 58,
  "updrs_part_iv": 9,
  "disease_duration_years": 8.0,
  "amyloid_beta_42_pg_ml": 690,
  "amyloid_beta_40_pg_ml": 6800,
  "p_tau181_pg_ml": 1.4,
  "t_tau_pg_ml": 180,
  "nfl_pg_ml": 29.0,
  "gfap_pg_ml": 130,
  "alpha_synuclein_pg_ml": 1350,
  "neuroinflam_score": 0.48,
  "tau_amyloid_ratio": 0.002
}
```

`amyloid_beta_40_pg_ml` is accepted in the dashboard. The app automatically derives
`amyloid_beta_42_40_ratio` for the trained model when both amyloid beta values are present.

## Notes

- Missing inputs are allowed; the model preprocessing pipeline imputes unavailable values.
- The dashboard calls `NeurologicalPredictionService` directly, so FastAPI does not need to be running for Gradio.
- FastAPI and Gradio can run at the same time on ports `8000` and `7860`.
- If a port is already busy, stop the old process or choose another port.
