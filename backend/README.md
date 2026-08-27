# NeuroRisk Research Platform — Backend

The backend is a FastAPI application that validates research inputs, serves several neurological-risk models, exposes precomputed EEG cohort artifacts, and coordinates asynchronous EEG upload inference.

> [!WARNING]
> The API is a research prototype, not a medical device or diagnostic service. Do not make clinical decisions from its output.

## Contents

- [Responsibilities](#responsibilities)
- [Architecture](#architecture)
- [Setup](#setup)
- [Configuration](#configuration)
- [Run the API](#run-the-api)
- [API reference](#api-reference)
- [Testing](#testing)
- [Model artifacts](#model-artifacts)
- [EEG maintenance scripts](#eeg-maintenance-scripts)
- [Production notes](#production-notes)
- [Troubleshooting](#troubleshooting)

## Responsibilities

The backend supports four related but distinct paths:

1. **General biomedical assessment** accepts the metadata-bound 24-key request and invokes separate disease and risk pipelines.
2. **Voice assessment** sends validated audio to the configured Gemini model for structured feature extraction, then calls the speech classifier.
3. **Legacy biomedical scoring** retains the former 14-field soft-voting endpoint for API compatibility.
4. **EEG assessment** serves precomputed cohort data without an ML runtime and optionally performs CPU TorchScript inference for uploaded EEGLAB recordings.

## Architecture

```text
app/main.py
|-- request-id and access-log middleware
|-- CORS middleware
`-- app/api/
    |-- routes/                 HTTP validation and error translation
    |-- schemas/                Pydantic contracts
    `-- services/
        |-- prediction_service               Keras speech inference
        |-- gemini_audio_service             voice feature extraction
        |-- neurological_prediction_service scikit-learn inference
        |-- eeg_cohort_service               Tier 1 precomputed data
        |-- eeg_job_service                  bounded in-memory EEG jobs
        |-- eeg_preprocessing                MNE filtering/ICA/epoching
        `-- eeg_inference_service            TorchScript EEG inference
```

The HTTP layer returns generic operational errors and logs internal details server-side. Valid incoming `X-Request-ID` values are propagated; otherwise the middleware generates a UUID.

## Setup

Commands in this guide assume the current directory is `backend/`. Running from this directory is important because imports use the top-level `app` package and `.env` is loaded from the working directory.

### Requirements

- Python 3.11 recommended, 64-bit
- pip
- Sufficient disk space for TensorFlow, PyTorch, MNE, and scientific Python wheels
- A Gemini API key only for `/voice-assessments/`

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Use `python -m pip` rather than a bare `pip` command so packages are installed into the currently active interpreter.

### CPU-only PyTorch

EEG inference is intentionally CPU-oriented. To force the CPU wheel:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

PyTorch is only required for live EEG uploads. Without it, cohort browsing and model-card routes remain available and `inference_available` is `false`.

## Configuration

Copy `.env.example` to `.env`, then change only the values required by the environment.

| Variable | Default | Required | Description |
| --- | --- | --- | --- |
| `APP_NAME` | `NeuroRisk Research Platform` | No | Shared application and OpenAPI/FastAPI title |
| `ENVIRONMENT` | `development` | No | Environment label |
| `LOG_LEVEL` | `INFO` | No | Logging threshold |
| `GEMINI_API_KEY` | empty | Voice only | Server-side Gemini credential |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | No | Gemini audio-analysis model |
| `FRONTEND_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | No | Comma-separated CORS origins |
| `EEG_JOB_TTL_SECONDS` | `1800` | No | Completed/failed job retention in seconds |
| `EEG_MAX_ACTIVE_JOBS` | `4` | No | Bounded active EEG-job count |

Example:

```dotenv
APP_NAME=NeuroRisk Research Platform
ENVIRONMENT=development
LOG_LEVEL=INFO
GEMINI_API_KEY=replace-with-your-server-side-api-key
GEMINI_MODEL=gemini-3.5-flash-lite
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
EEG_JOB_TTL_SECONDS=1800
EEG_MAX_ACTIVE_JOBS=4
```

`.env` is ignored by Git. Do not expose `GEMINI_API_KEY` to the frontend.

## Run the API

Development:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Direct module form, which guarantees use of the active environment:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/health/
```

Useful URLs:

- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

## API reference

| Method | Path | Purpose | Special dependency |
| --- | --- | --- | --- |
| `GET` | `/health/` | Liveness check | None |
| `POST` | `/predictions/` | Legacy 14-field biomedical scoring | 130-feature preprocessor plus soft-voting ensemble |
| `POST` | `/voice-assessments/` | Extract features from audio and score them | Gemini API key plus speech model |
| `POST` | `/neurological-risk/predict` | Score the General 24-field biomedical contract | Separate saved disease and risk pipelines |
| `GET` | `/eeg/model-card` | Model architecture, performance, intended use, confounds | Installed JSON model card |
| `GET` | `/eeg/band-reference` | Cohort band-power context | Precomputed cohort artifacts |
| `GET` | `/eeg/cohort` | Paginated/filterable cohort list | Precomputed cohort artifacts |
| `GET` | `/eeg/cohort/projection` | 2-D cohort embedding projection | Precomputed cohort artifacts |
| `GET` | `/eeg/cohort/{subject_id}` | Complete subject report | Precomputed cohort artifacts |
| `GET` | `/eeg/embeddings/{subject_id}` | Full 256-D `z_eeg` vector | Precomputed cohort artifacts |
| `POST` | `/eeg/assessments` | Submit `.set` and optional `.fdt`; returns HTTP 202 | PyTorch, MNE, TorchScript graph |
| `GET` | `/eeg/assessments/{job_id}` | Poll an EEG job | Existing in-memory job |

### General prediction example

The endpoint requires exactly the 24 keys below and rejects missing or unexpected keys. Any value may be `null`; the fitted numeric median or categorical mode imputer then handles it. The disease pipeline returns AD/Healthy/MS/PD probabilities. The separate risk pipeline returns Low/Medium/High plus `0×P(Low) + 50×P(Medium) + 100×P(High)`, normalized to the API's 0–1 `risk_score`.

```bash
curl -X POST http://127.0.0.1:8000/neurological-risk/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: readme-example" \
  -d '{
    "age": 65,
    "sex": "Female",
    "education_years": 14,
    "bmi": 25.0,
    "family_history_pd": 0,
    "systolic_bp": 125,
    "diastolic_bp": 80,
    "cognitive_screen_score_0_30": 27,
    "rem_sleep_score": 4,
    "updrs_part_i": 4,
    "updrs_part_ii": 5,
    "updrs_part_iii": 12,
    "updrs_part_iv": 0,
    "schwab_england_adl": 90,
    "apoe_e4_count": 1,
    "gba_variant_carrier": 0,
    "amyloid_beta_42_40_ratio": 0.08,
    "t_tau_pg_ml": 300.0,
    "p_tau181_pg_ml": 50.0,
    "nfl_pg_ml": 800.0,
    "gfap_pg_ml": 200.0,
    "alpha_synuclein_pg_ml": 1200.0,
    "gdf15_pg_ml": 800.0,
    "crp40_copy_number": 2000.0
  }'
```

### Voice assessment example

Allowed MIME formats are WAV, MP3, MP4/M4A, OGG, and WebM. Files must be 18 MB or smaller, age must be 18–120, and `recording_task` must be `reading`, `monologue`, `picture_description`, or `sustained_vowel`.

```bash
curl -X POST http://127.0.0.1:8000/voice-assessments/ \
  -F "file=@sample.wav;type=audio/wav" \
  -F "patient_age=65" \
  -F "recording_task=reading"
```

### EEG cohort examples

```bash
curl http://127.0.0.1:8000/eeg/model-card
curl "http://127.0.0.1:8000/eeg/cohort?true_class=AD&site=AR&offset=0&limit=20"
curl http://127.0.0.1:8000/eeg/cohort/AD-AR-sub-30001
curl http://127.0.0.1:8000/eeg/embeddings/AD-AR-sub-30001
```

Valid cohort filters are:

- `true_class`: `AD`, `PD`, `MS`, or `HC`
- `site`: `AR` or `CL`
- `quality`: `Good`, `Moderate`, or `Poor`
- `offset`: zero or greater
- `limit`: 1–200

### EEG upload and polling

Self-contained `.set`:

```bash
curl -X POST http://127.0.0.1:8000/eeg/assessments \
  -F "files=@subject.set"
```

External data pair:

```bash
curl -X POST http://127.0.0.1:8000/eeg/assessments \
  -F "files=@subject.set" \
  -F "files=@subject.fdt"
```

The response has HTTP status 202 and contains a `job_id`. Poll until `status` becomes `completed` or `failed`:

```bash
curl http://127.0.0.1:8000/eeg/assessments/JOB_ID
```

Jobs move through `queued`, `validating`, `preprocessing`, `inference`, then `completed` or `failed`. The combined upload limit is 120 MB. A `.set` header that refers to external signal data must be uploaded with its matching `.fdt` file.

## Testing

### Full suite

Activate `.venv`, remain in `backend/`, and run:

```bash
python -m pytest
```

### Common pytest commands

```bash
# Concise output
python -m pytest -q

# Stop after the first failure
python -m pytest -x

# One file
python -m pytest tests/test_api.py -v

# One test
python -m pytest tests/test_health.py::test_app_imports -v

# Tests matching an expression
python -m pytest -k "eeg and not parity" -v
```

The suite covers request validation, error redaction, request IDs, speech inference composition, voice uploads, neurological probabilities, EEG cohort integrity, model-card disclosures, preprocessing, job limits/expiry, band statistics, and serving parity.

Verified baseline when this guide was written:

```text
collected 91 items
90 passed, 1 skipped
```

The skipped parity case depends on optional runtime/model conditions and is reported by pytest rather than hidden.

### Smoke-test a running server

With Uvicorn running in another terminal:

```bash
python scripts/test_backend.py
```

Custom base URL:

```bash
python scripts/test_backend.py --url http://127.0.0.1:9000
```

The smoke test uses only the Python standard library and checks `/health/`, `/predictions/`, the request ID, and the main response fields.

## Model artifacts

The application expects committed artifacts at fixed paths.

| Path | Used by |
| --- | --- |
| `app/models/speech_neuro_risk_classifier.keras` | General and voice speech classifier |
| `app/models/feature_scaler.joblib` | Speech feature scaling |
| `app/models/label_encoder.joblib` | Speech class labels |
| `app/models/feature_columns.joblib` | Trained speech feature order |
| `app/models/neurological_risk_model.joblib` | General disease and risk inference at `/neurological-risk/predict` |
| `app/models/neurological_risk_model_metadata.json` | Exact run, schema, classes, metrics, and runtime contract |
| `app/models/eeg/model_card.json` | EEG architecture, metrics, disclosures, availability |
| `app/models/eeg/neuro_risk_encoder.torchscript.pt` | Live EEG CPU inference |
| `app/models/eeg/neuro_risk_inference_bundle.joblib` | EEG preprocessing and output contract |
| `app/models/eeg/cohort/` | Cohort index, reports, embeddings, projection, band reference |

Do not rename these files without updating their loader services. See [the EEG bundle README](app/models/eeg/README.md) for the installed run's scientific and serving caveats.

## EEG maintenance scripts

Run scripts from `backend/` with the environment activated.

| Script | Purpose |
| --- | --- |
| `scripts/test_backend.py` | Smoke-test a running API |
| `scripts/verify_eeg_bundle.py` | Validate the installed model card, bundle, reports, and related invariants |
| `scripts/check_serving_parity.py` | Compare serving behavior with exported run expectations |
| `scripts/finalize_eeg_model.py` | Install/finalize an exported EEG training run |
| `scripts/build_eeg_cohort_index.py` | Build the cohort index; `--fixtures` creates synthetic development data |
| `scripts/backfill_eeg_embeddings.py` | Populate missing cohort embeddings |
| `scripts/build_neurological_runtime_model.py` | Build the neurological runtime model artifact |

After replacing an EEG model, the minimum checks are:

```bash
python scripts/verify_eeg_bundle.py
python scripts/check_serving_parity.py
python -m pytest
```

## Production notes

- Terminate TLS at a trusted reverse proxy or managed platform.
- Use an explicit CORS allowlist; never use a wildcard for a credentialed or sensitive deployment.
- Keep one Uvicorn worker with the current EEG job implementation. The registry is process-local, so a job submitted to one worker may not be visible when a poll reaches another worker.
- For multi-worker or multi-instance deployment, replace the in-memory job registry with a durable shared queue and result store.
- Set upload/body limits at the reverse proxy consistently with the backend's 18 MB voice and 120 MB EEG limits.
- Restrict logs and avoid recording request bodies or participant identifiers.
- Store secrets in the deployment secret manager rather than a committed file.
- Add authentication, authorization, persistence policy, audit controls, and compliance review before any use with real patient data.

Single-process production-style command:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Change to `backend/` before starting Uvicorn, and confirm the virtual environment is active.

### Prediction returns HTTP 503

Confirm the model artifacts are present and TensorFlow/Keras, joblib, scikit-learn 1.8.0, and the remaining requirements are installed in the active environment.

### Voice returns HTTP 503

Set `GEMINI_API_KEY` in `.env`, confirm `google-genai` is installed, and restart the server. Also confirm the server can reach the Gemini API.

### EEG upload returns HTTP 503

Request `/eeg/model-card` and inspect `inference_available`. Install CPU PyTorch if it is false and confirm the TorchScript graph is present.

### EEG upload returns HTTP 422

Check that the upload contains exactly one valid `.set`, includes the required `.fdt` companion, uses the expected BioSemi 128-channel montage, is long enough to produce at least 20 clean epochs, and meets the signal-quality thresholds.

### Browser reports a CORS error

Add the frontend's exact scheme, host, and port to `FRONTEND_ORIGINS`; then restart Uvicorn. CORS origins do not include URL paths.
