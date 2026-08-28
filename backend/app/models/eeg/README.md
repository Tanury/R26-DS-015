# EEG model bundle

What the API serves for `/eeg/*`.

```
model_card.json                     architecture, metrics, confound disclosure
neuro_risk_encoder.torchscript.pt   the served graph (torch.jit.load, no project code)
neuro_risk_inference_bundle.joblib  channel order, bands, thresholds, centroids
neuro_risk_encoder.onnx             cross-framework copy (not used by the API)
cohort/index.json                   one row per assessed subject
cohort/{subject_id}.json            per-subject report
cohort/embeddings/{subject_id}.json 256-D z_eeg vector
cohort/projection.json              2-D PCA of the cohort embeddings
cohort/band_reference.json          per-class band power + whether it separates
```

## Currently installed: a real training run

`run_id 20260820T114437Z` — **Lightweight EEG Transformer** on the **raw** time-domain
tensor `[1, 128, 1024]`, the configuration that won both ablations. 115 BrainLat
subjects (AD 35 · HC 32 · PD 26 · MS 22), subject-level 5-fold CV.

Pooled held-out AUC: AD 0.967, PD 0.988, MS 0.992; binary neuro-vs-HC 0.982.

Installed with:

```bash
python scripts/finalize_eeg_model.py --run ../neuro-ai-eeg
python scripts/verify_eeg_bundle.py
python scripts/check_serving_parity.py
```

## Two things about this run you need to know

**The age probe never ran.** The notebook's demographics merge produced an empty
`age` column, so `confound_probes.json` records `age: {n: 0}` and the age-matched
sub-analysis is empty. Age is the dominant confound on this cohort — MS
participants are ~33 years younger than every other group — so `finalize_eeg_model.py`
recovers ages from the dataset demographics afterwards and correlates each
delivered risk score against them. Measured on 87 of 115 subjects:

| Condition | r vs age (all) | r vs age (within non-cases) |
|---|---:|---:|
| AD | +0.54 | **+0.10** |
| PD | −0.24 | −0.24 (PD ages unrecoverable) |
| MS | −0.77 | **−0.15** |

The within-non-cases column is the decisive one: among subjects who do *not* have
the condition, the score barely tracks age. That is the evidence the heads are not
simply age detectors, and it is why MS is graded `HIGH (site)` rather than
`CRITICAL (age + site)`. PD demographics are an HTML error page, so the PD score
cannot be age-assessed at all.

**Serving must use `legacy_eps` standardization.** This model was trained before the
per-epoch z-score epsilon fix, so it was fitted on `x / (std + 1e-6)`. Serving it
under exact z-scoring moves a held-out healthy control from PD 0.026 to **PD 0.848**.
The bundle therefore carries a `standardization` field (defaulting to `legacy_eps`
for bundles that predate the fix), and `scripts/check_serving_parity.py` verifies
serving reproduces the run on real recordings — worst deviation 0.025 across eight
subjects spanning all four classes.

Retraining is what switches a deployment to `zscore_exact`: the notebooks now use
the corrected z-score and stamp `"standardization": "zscore_exact"` into the bundle
they export.

## Replacing this model

```bash
python scripts/finalize_eeg_model.py --run <dir with r26_ds015_model/ and r26_ds015_artifacts/>
python scripts/verify_eeg_bundle.py
python scripts/check_serving_parity.py     # always run this after a swap
```

`build_eeg_cohort_index.py --fixtures` regenerates a synthetic development cohort
if you need one; every fixture report carries `"fixture": true`.

## The band reference, and why MS comes back empty

`cohort/band_reference.json` (served at `/eeg/band-reference`) holds per-class band
medians and a rank-based AUC against controls, so a single subject's spectrum can be
placed against the group it came from. It is **descriptive**: the encoder consumes raw
time-domain epochs and never sees band power, so this says "your recording resembles
the AD group's recordings", not "this drove your score". Occlusion is the only causal
attribution in the bundle.

|  | AD | PD | MS |
|---|---:|---:|---:|
| theta/alpha AUC vs HC | 0.72 | 0.84 | **0.48** |
| bands separating | 3 of 6 | 5 of 6 | 1 of 6, reversed |
| `has_signature` | true | true | **false** |

MS sits at chance on every canonical slowing axis while the encoder separates it at
0.992, and its one separating band (delta) runs backwards — MS participants here are
~33 years younger, and younger brains show less slow-wave activity. `has_signature`
therefore has to be able to return False, and the UI states the absence rather than
charting one. The rule lives in `services/eeg_band_statistics.py`; `verify_eeg_bundle.py`
warns if a future run reports a signature for every condition.

Full analysis, including what would fix each condition:
[`EEG_BAND_PATTERN_FINDINGS.md`](../../../../EEG_BAND_PATTERN_FINDINGS.md).

## Coverage caveat

The training run exports a *full* deep report for only its demo subjects (one per
class), so per-subject embedding geometry and occlusion explainability exist for 4
of 115. Every other subject has real risk scores, signal quality and band power,
with those two blocks marked unavailable rather than invented — the report's
`signal_quality.warnings` says so per subject.

## Rejected uploads

Two subjects in this cohort fail preprocessing and are excluded — the same two the
training run excluded:

| Subject | Why |
|---|---|
| `PD-AR-sub-40010` | 0 of 192 epochs pass 150 µV; typical epoch ~690 µV, B5 at 1,348 µV |
| `MS-AR-suj_512` | 5 of 228 epochs pass — below the 20-epoch minimum |

Uploading either returns a failed job with a structured `details` block naming the
offending electrodes, because amplitude rejection drops an epoch when its *worst*
channel exceeds the limit — a couple of detached electrodes can fail an otherwise
usable recording. The UI renders that as a per-channel chart.

## Tier 2 (upload + inference)

Requires PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Without it, `/eeg/model-card` reports `inference_available: false`, uploads return
503 with an explanation, and cohort browsing is unaffected.
