import type {
  BandReference,
  CohortPage,
  CohortProjection,
  EegJob,
  EegModelCard,
  EegRiskReport,
} from "@/lib/eeg-types";

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

export async function fetchModelCard() {
  return parseResponse<EegModelCard>(await fetch(`${API_BASE_URL}/eeg/model-card`));
}

export async function fetchCohort(params: {
  trueClass?: string;
  site?: string;
  quality?: string;
  offset?: number;
  limit?: number;
} = {}) {
  const query = new URLSearchParams();
  if (params.trueClass) query.set("true_class", params.trueClass);
  if (params.site) query.set("site", params.site);
  if (params.quality) query.set("quality", params.quality);
  query.set("offset", String(params.offset ?? 0));
  query.set("limit", String(params.limit ?? 50));
  return parseResponse<CohortPage>(await fetch(`${API_BASE_URL}/eeg/cohort?${query}`));
}

export async function fetchCohortReport(subjectId: string) {
  return parseResponse<EegRiskReport>(
    await fetch(`${API_BASE_URL}/eeg/cohort/${encodeURIComponent(subjectId)}`),
  );
}

export async function fetchBandReference() {
  return parseResponse<BandReference>(await fetch(`${API_BASE_URL}/eeg/band-reference`));
}

export async function fetchProjection() {
  return parseResponse<CohortProjection>(
    await fetch(`${API_BASE_URL}/eeg/cohort/projection`),
  );
}

export async function submitEegAssessment(files: File[]) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  return parseResponse<EegJob>(
    await fetch(`${API_BASE_URL}/eeg/assessments`, { method: "POST", body: form }),
  );
}

export async function fetchEegJob(jobId: string) {
  return parseResponse<EegJob>(
    await fetch(`${API_BASE_URL}/eeg/assessments/${encodeURIComponent(jobId)}`),
  );
}

/**
 * Poll a job until it settles. Preprocessing runs 30-90 s, so a 2 s interval keeps
 * the progress bar responsive without hammering the API.
 */
export async function pollEegJob(
  jobId: string,
  onUpdate: (job: EegJob) => void,
  options: { intervalMs?: number; timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<EegJob> {
  const interval = options.intervalMs ?? 2000;
  const timeout = options.timeoutMs ?? 10 * 60 * 1000;
  const startedAt = Date.now();

  for (;;) {
    if (options.signal?.aborted) throw new Error("Polling cancelled.");
    const job = await fetchEegJob(jobId);
    onUpdate(job);
    if (job.status === "completed" || job.status === "failed") return job;
    if (Date.now() - startedAt > timeout) {
      throw new Error("The assessment is taking longer than expected. Check back shortly.");
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
}
