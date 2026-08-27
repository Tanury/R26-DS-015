import type { BiomedicalFeatures, Prediction, VoiceAssessment } from "@/lib/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "Request failed.";
    throw new Error(detail);
  }
  return body as T;
}

export async function submitGeneralAssessment(features: BiomedicalFeatures) {
  const response = await fetch(`${API_BASE_URL}/neurological-risk/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
  return parseResponse<Prediction>(response);
}

export async function submitVoiceAssessment(
  file: File,
  patientAge: number,
  recordingTask: string,
) {
  const form = new FormData();
  form.append("file", file);
  form.append("patient_age", String(patientAge));
  form.append("recording_task", recordingTask);
  const response = await fetch(`${API_BASE_URL}/voice-assessments/`, {
    method: "POST",
    body: form,
  });
  return parseResponse<VoiceAssessment>(response);
}
