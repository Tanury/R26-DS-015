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