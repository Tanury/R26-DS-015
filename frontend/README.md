# NeuroRisk Research Platform — Frontend

The frontend is a Next.js App Router application for general speech-biomarker assessment, voice upload, EEG cohort exploration, EEG upload progress, results, history, insights, and model-card disclosure.

> [!WARNING]
> The interface presents research indicators, not medical diagnoses. Keep the disclaimer visible and do not repurpose these views for clinical decision-making without validation and appropriate governance.

## Stack

- Next.js 16.3.1 with the App Router and Turbopack
- React 19.2
- TypeScript 5.9
- Tailwind CSS 4
- Radix UI Slot and local UI primitives
- Lucide React icons
- Browser `fetch`, `localStorage`, and `sessionStorage`

## Requirements

- Node.js 20.9 or newer, as required by the installed Next.js package
- Node.js 22 LTS recommended
- npm 10 or newer
- The FastAPI backend running locally or at a reachable deployment URL

## Setup

Commands in this guide assume the current directory is `frontend/`.

### 1. Install exact locked dependencies

```bash
npm ci
```

Use `npm ci` for a fresh clone and CI because it installs from `package-lock.json` without rewriting the lockfile. Use `npm install <package>` only when intentionally changing dependencies.

### 2. Create local configuration

macOS or Linux:

```bash
cp .env.local.example .env.local
```

Windows PowerShell:

```powershell
Copy-Item .env.local.example .env.local
```

Default development configuration:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

The API client removes one trailing slash automatically. Supply only the origin/base URL, with no `/docs` or endpoint path.

> [!CAUTION]
> Every `NEXT_PUBLIC_*` value is bundled into browser JavaScript. Never store the Gemini key or any other secret in a frontend environment variable.

### 3. Start development mode

```bash
npm run dev
```

Open <http://localhost:3000>.

The backend must allow the exact frontend origin through `FRONTEND_ORIGINS`. The default backend configuration allows both `http://localhost:3000` and `http://127.0.0.1:3000`.

## Available scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Next.js development server |
| `npm run lint` | Run ESLint across the frontend |
| `npm run build` | Run TypeScript checks and create an optimized production build |
| `npm run start` | Serve the existing production build |

## Routes

| Route | Purpose | Backend dependency |
| --- | --- | --- |
| `/` | Dashboard and entry points | None for initial shell |
| `/general` | Enter the exact 24 clinical/genetic/biomarker fields | `POST /neurological-risk/predict` on submit |
| `/general/results` | Display disease probabilities, independent risk, explanation, and suggestions | Browser session storage |
| `/voice` | Record or upload audio | `POST /voice-assessments/` |
| `/voice/results` | Display transcript, extracted features, and prediction | Browser session storage |
| `/eeg` | Featured cohort subjects and EEG upload | EEG model card, cohort, and assessment routes |
| `/eeg/cohort` | Browse/filter cohort reports | EEG cohort/projection routes |
| `/eeg/results` | EEG risk, quality, bands, embedding, and confounds | Browser session storage; band reference as needed |
| `/model-card` | Installed EEG model performance and disclosures | `GET /eeg/model-card` |
| `/history` | Up to 100 locally saved results | Browser local storage |
| `/insights` | Research summary/insight views | Local application data |
| `/settings` | Local UI/settings page | None |
| `/support` | Usage guidance | None |

## Frontend data flow

```text
Page/form
  |
  v
lib/api.ts or lib/eeg-api.ts
  |
  v
NEXT_PUBLIC_API_BASE_URL
  |
  v
FastAPI JSON or multipart endpoint
  |
  v
typed result -> browser history -> result route
```

Speech and EEG result history is intentionally client-side:

- `localStorage["neurorisk-assessments"]` stores up to 100 history entries.
- `sessionStorage["neurorisk-current-result"]` stores the result shown by a results page.
- There is no login, synchronization, or server-side history database.
- Clearing site data removes the saved history.

This storage is convenient for a prototype but is not an approved clinical record or secure participant-data store.

## Backend integration

The frontend has two API client modules:

- `lib/api.ts` handles general feature and voice assessment requests.
- `lib/eeg-api.ts` handles model-card, cohort, report, band-reference, projection, upload, and job-polling requests.

Both default to `http://127.0.0.1:8000` when `NEXT_PUBLIC_API_BASE_URL` is absent. Explicit configuration is still recommended so deployment behavior is obvious.

EEG uploads return a job immediately. The browser polls every two seconds until the job is `completed` or `failed`, with a default client-side timeout of ten minutes. Normal preprocessing is expected to take approximately 30–90 seconds.

## Quality checks

Run both checks before opening a pull request or producing a release build:

```bash
npm run lint
npm run build
```

The production build performs TypeScript validation and statically renders the App Router pages. There is currently no Jest, Vitest, or browser end-to-end test script configured in `package.json`; lint and build are the available automated frontend gates.

Verified baseline when this guide was written:

```text
npm run build: successful
npm run lint: no errors; one existing unused-import warning
```

The warning is in `components/research-disclaimer.tsx` and does not fail the configured lint command.

## Production build and run

Set the public backend URL before building because public environment values may be embedded into the client bundle:

```dotenv
NEXT_PUBLIC_API_BASE_URL=https://api.example.org
```

Then run:

```bash
npm ci
npm run lint
npm run build
npm run start
```

`npm run start` serves on port 3000 by default. To select another port:

macOS or Linux:

```bash
PORT=3001 npm run start
```

Windows PowerShell:

```powershell
$env:PORT=3001
npm run start
```

When the public frontend origin changes, add it to the backend's `FRONTEND_ORIGINS` value.

## Manual verification checklist

With both services running:

1. Open `/` and confirm navigation renders at desktop and narrow viewport widths.
2. Open `/general`, enter available values, submit, and confirm navigation to `/general/results`; verify blank fields are shown as not supplied.
3. Open `/voice`, verify file validation and recording controls, and submit only when Gemini is configured.
4. Open `/eeg`, confirm the model summary and featured cohort list load.
5. Open `/eeg/cohort`, apply class/site/quality filters, and open a subject.
6. Confirm `/eeg/results` labels the three risk scores as independent and displays confound disclosures.
7. Open `/model-card` and confirm metrics/intended-use sections load.
8. Open `/history`, refresh the page, and confirm locally stored results remain.
9. Inspect the browser console and Network panel for runtime, CORS, or failed-request errors.

## Project structure

```text
frontend/
|-- app/
|   |-- eeg/                 # upload, cohort, and EEG results
|   |-- general/             # manual feature input and results
|   |-- voice/               # audio input and results
|   |-- layout.tsx           # root layout
|   `-- globals.css          # Tailwind and shared styles
|-- components/
|   |-- ui/                  # button, card, and input primitives
|   `-- *.tsx                # domain and layout components
|-- lib/
|   |-- api.ts               # speech/voice API client
|   |-- eeg-api.ts           # EEG API client and polling
|   |-- history.ts           # local/session storage
|   |-- types.ts             # speech UI contracts
|   `-- eeg-types.ts         # EEG UI contracts
|-- .env.local.example
|-- next.config.ts
|-- package.json
`-- package-lock.json
```

## Troubleshooting

### `npm ci` fails because the lockfile and manifest differ

Do not delete the lockfile as a first response. On the branch where dependencies were intentionally changed, run `npm install`, review the resulting `package-lock.json` diff, then commit both manifest and lockfile together.

### The app displays “API is not reachable”

- Start the backend and check <http://127.0.0.1:8000/health/>.
- Confirm `NEXT_PUBLIC_API_BASE_URL` is correct.
- Restart `npm run dev` after editing `.env.local`.
- Check the browser Network panel for the requested URL and response.

### The browser reports a CORS failure

Set the backend's `FRONTEND_ORIGINS` to the exact frontend origin, including scheme and port, then restart the backend.

### General assessment returns 422

The request must contain all 24 keys and no unexpected names. Leave unavailable form values blank; the frontend sends those keys as `null` for fitted imputation. Correct any supplied value shown outside its accepted range.

### Voice assessment returns 503

The backend needs a valid server-side `GEMINI_API_KEY`. This is not a frontend credential and must never be added to `.env.local`.

### EEG cohort loads but upload is disabled

The model card reported `inference_available: false`. Install CPU PyTorch in the backend environment and confirm the TorchScript artifact is installed. Cohort exploration is designed to keep working without live inference.

### A results page is empty after opening it directly

Result pages read the current assessment from `sessionStorage`. Submit or open an assessment in the same browser tab before visiting a results route.
