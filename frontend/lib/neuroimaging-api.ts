import type { HistoryItem, NeuroimagingResult } from "@/lib/types";

// Vision Encoder API client — mirrors the pattern used in lib/eeg-api.ts
// (env-driven base URL instead of a hardcoded host, so this works the same
// way locally and once deployed).

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

async function parseOrThrow(res: Response) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed.");
  }
  return res.json();
}

export async function analyzeMRI(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/analyze/mri`, {
    method: "POST",
    body: form,
  });
  return parseOrThrow(res);
}

export async function preprocessDICOM(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/preprocess/dicom`, {
    method: "POST",
    body: form,
  });
  return parseOrThrow(res);
}

export async function analyzeDatPipeline(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/pd/analyze/dat/pipeline`, {
    method: "POST",
    body: form,
  });
  return parseOrThrow(res);
}

export async function analyzePdMRI(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/pd/analyze/mri`, {
    method: "POST",
    body: form,
  });
  return parseOrThrow(res);
}

export async function analyzeMsMRI(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/ms/analyze/mri`, {
    method: "POST",
    body: form,
  });
  return parseOrThrow(res);
}

export async function analyzeMsOct(volFile: File, matFile: File) {
  const form = new FormData();
  form.append("vol_file", volFile);
  form.append("mat_file", matFile);
  const res = await fetch(`${API_BASE_URL}/api/ms/analyze/oct`, {
    method: "POST",
    body: form,
  });
  return parseOrThrow(res);
}
const LABEL_ALIASES: Record<string, string> = {
  "multiple sclerosis": "MS",
  "healthy control": "HC",
  "parkinson's disease": "PD",
  "alzheimer's disease": "AD",
  "mild cognitive impairment": "MCI",
  "cognitively normal": "CN",
};

type ImagingPrediction = {
  prediction?: string;
  confidence?: number;
  risk_level?: string;
  modality?: string;
  filename?: string;
  class_scores?: { label: string; confidence: number }[];
  model_version?: string;
  error?: string;
};

/**
 * Normalise an imaging endpoint response into the record the fusion engine
 * reads. Returns null when inference failed or produced no class scores.
 */
export function toNeuroimagingResult(
  data: ImagingPrediction | null | undefined,
  condition: NeuroimagingResult["condition"],
  scanType: string,
  filename?: string,
): NeuroimagingResult | null {
  if (!data || data.error || !data.class_scores?.length) return null;
  const probabilities = Object.fromEntries(
    data.class_scores.map(({ label, confidence }) => [
      LABEL_ALIASES[label.trim().toLowerCase()] ?? label.trim().toUpperCase(),
      Number(confidence),
    ]),
  );
  const prediction = data.prediction ?? "";
  return {
    condition,
    scanType,
    modality: data.modality ?? scanType,
    filename: data.filename ?? filename,
    prediction: LABEL_ALIASES[prediction.trim().toLowerCase()] ?? prediction,
    confidence: Number(data.confidence ?? 0),
    riskLevel: data.risk_level ?? "UNKNOWN",
    modelVersion: data.model_version,
    demo: /demo|mock/i.test(data.model_version ?? ""),
    probabilities,
  };
}

type CohortSubject = {
  subject_id: string;
  condition: NeuroimagingResult["condition"];
  scan_type: string;
  modality: string;
  true_label: string;
  prediction: string;
  confidence: number;
  risk_level: string;
  probabilities: Record<string, number>;
  embedding_dim: number;
  in_sample: boolean;
  model_metrics: { loocv_accuracy: number; loocv_roc_auc: number };
};

export type ImagingCohort = {
  generated_at: string;
  conditions_available: NeuroimagingResult["condition"][];
  conditions_missing: Partial<Record<NeuroimagingResult["condition"], string>>;
  items: HistoryItem[];
};

/**
 * Vision-encoder results precomputed over the neuroimaging module's processed
 * features, shaped as history items so fusion treats them like saved scans.
 */
export async function fetchImagingCohort(): Promise<ImagingCohort> {
  const res = await fetch(`${API_BASE_URL}/api/imaging/cohort`);
  const data = await parseOrThrow(res);
  const items: HistoryItem[] = (data.subjects as CohortSubject[]).map((subject) => ({
    id: subject.subject_id,
    createdAt: data.generated_at,
    type: "Neuroimaging",
    neuroimaging: {
      condition: subject.condition,
      scanType: subject.scan_type,
      modality: subject.modality,
      prediction: subject.prediction,
      confidence: subject.confidence,
      riskLevel: subject.risk_level,
      probabilities: subject.probabilities,
      cohort: {
        trueLabel: subject.true_label,
        inSample: subject.in_sample,
        embeddingDim: subject.embedding_dim,
        loocvAccuracy: subject.model_metrics.loocv_accuracy,
        loocvAuc: subject.model_metrics.loocv_roc_auc,
      },
    },
  }));
  return {
    generated_at: data.generated_at,
    conditions_available: data.conditions_available,
    conditions_missing: data.conditions_missing,
    items,
  };
}

export type UploadCondition = NeuroimagingResult["condition"];

export type ImagingUploadMode = {
  id: string;
  label: string;
  scanType: string;
  /** Every file the backend needs for this scan type. */
  files: { key: string; label: string; accept: string }[];
};

/**
 * Scan types the fusion page accepts for local upload. AD's DICOM pipeline is
 * left out because it needs FSL (bet/flirt), which does not run on Windows.
 */
export const IMAGING_UPLOAD_MODES: Record<UploadCondition, ImagingUploadMode[]> = {
  AD: [
    { id: "nifti", label: "Structural MRI (.nii / .nii.gz)", scanType: "Structural MRI (NIfTI)", files: [{ key: "file", label: "NIfTI scan", accept: ".nii,.nii.gz,.gz" }] },
  ],
  PD: [
    { id: "dat", label: "DaTscan SPECT (.dcm)", scanType: "DaTscan SPECT", files: [{ key: "file", label: "DaTscan DICOM", accept: ".dcm,.ima,.img" }] },
    { id: "nifti", label: "Brain MRI (.nii / .nii.gz)", scanType: "Structural MRI (NIfTI)", files: [{ key: "file", label: "NIfTI scan", accept: ".nii,.nii.gz,.gz" }] },
  ],
  MS: [
    { id: "oct", label: "Retinal OCT (.vol + .mat)", scanType: "Retinal OCT", files: [{ key: "vol", label: "OCT volume (.vol)", accept: ".vol" }, { key: "mat", label: "Segmentation (.mat)", accept: ".mat" }] },
    { id: "mri", label: "Brain MRI (.nii / .nii.gz)", scanType: "Brain MRI (MindGlide)", files: [{ key: "file", label: "NIfTI scan", accept: ".nii,.nii.gz,.gz" }] },
  ],
};

/** Run one imaging model on local files and normalise its output for fusion. */
export async function runImagingUpload(
  condition: UploadCondition,
  modeId: string,
  files: Record<string, File>,
): Promise<NeuroimagingResult> {
  const mode = IMAGING_UPLOAD_MODES[condition].find((candidate) => candidate.id === modeId);
  if (!mode) throw new Error(`Unknown ${condition} scan type: ${modeId}`);
  let data: ImagingPrediction;
  if (condition === "MS" && modeId === "oct") data = await analyzeMsOct(files.vol, files.mat);
  else if (condition === "MS") data = await analyzeMsMRI(files.file);
  else if (condition === "PD" && modeId === "dat") data = (await analyzeDatPipeline(files.file)).prediction;
  else if (condition === "PD") data = await analyzePdMRI(files.file);
  else data = await analyzeMRI(files.file);
  const result = toNeuroimagingResult(data, condition, mode.scanType, (files.file ?? files.vol)?.name);
  if (!result) throw new Error(data?.error ? `Inference failed: ${data.error}` : "The imaging model returned no class scores.");
  return result;
}
