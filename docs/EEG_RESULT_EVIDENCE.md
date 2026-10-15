# EEG result evidence

The upload flow remains `/eeg` → assessment job → `/eeg/prediction`.
Prediction and detailed-result pages now share evidence and sample-comparison panels.

## Scores and selection

- Four-class scores are read from the auxiliary model output, never generated for display.
- A complete, finite four-class distribution is ranked directly. The independent AD,
  PD and MS risk scores remain separately visible.
- If the auxiliary distribution is unavailable, rank the measured independent risk
  heads. There is no inferred healthy-control score.
- Exact ties have no selected class. Missing values remain unavailable. Small score
  margins, multiple elevated risk heads, quality warnings and model confounds are disclosed.
- Explanations describe score gaps and available occlusion/centroid observations.
  New occlusion results explicitly identify their target head and condition. Older
  reports without that provenance cannot establish an attribution of the selection.

## Measured waveform

Preprocessing retains the first selected clean epoch at A1/B1/C1/D1, after filtering
and artifact removal and before standardization. The samples are converted from volts
to microvolts for the report. This adds no transformation to the model input.
The UI uses a shared amplitude scale and time axis. Existing saved reports without
samples show an explicit unavailable state; no waveform is synthesized.

The short excerpt persists with the report in existing browser assessment storage.
The original upload files still follow the existing temporary-file cleanup workflow.

## Similar reference examples

`POST /eeg/similar-samples` accepts a subject ID, model run ID and the five relative
band powers. It returns the nearest real sample from each available cohort class,
ranked by root mean square band-power difference. No predicted-label filtering or
artificial similarity percentage is used. The displayed difference is in percentage
points; a nearest sample need not be a close match.

The search excludes the same subject ID, synthetic fixtures, incomplete profiles
and other model runs. Older reports recover a missing run ID only when their export
timestamp, architecture, representation and embedding dimension match the model card.
Future cohort exports stamp the run ID directly. Reference examples are not matched
for age or site and are descriptive, not a diagnostic validation or a model attribution.

`/eeg/cohort/[subjectId]` opens a reference report without replacing the user's saved
assessment. Request failures have a retry action; empty comparisons remain explicit.

## Verification

- `node --test tests/eeg-prediction-guidance.test.cjs` from `frontend`.
- `python -m pytest tests/test_eeg_evidence.py` from `backend`, plus existing EEG suites.
- The inference integration test uses explicitly synthetic signal input to exercise
  cleaning, the deployed encoder, waveform serialization and occlusion provenance;
  it makes no clinical accuracy claim.
