import { extractBiomedicalRecord, submitGeneralAssessment, submitVoiceAssessment } from "@/lib/api";
import { pollEegJob, submitEegAssessment } from "@/lib/eeg-api";
import type { EegJob } from "@/lib/eeg-types";
import { runImagingUpload, type UploadCondition } from "@/lib/neuroimaging-api";
import type { HistoryItem } from "@/lib/types";

/**
 * Uploads made directly on the fusion page run the same models as the
 * individual assessment pages, but the results live only in the fusion
 * session: nothing is written to assessment history.
 */
function fusionItem(prefix: string, fields: Omit<HistoryItem, "id" | "createdAt">): HistoryItem {
  return {
    id: `FUS-${prefix}-${String(Date.now()).slice(-6)}`,
    createdAt: new Date().toISOString(),
    ...fields,
  };
}

export async function runVoiceUpload(file: File, patientAge: number, recordingTask: string): Promise<HistoryItem> {
  if (!Number.isInteger(patientAge) || patientAge < 18 || patientAge > 120) {
    throw new Error("Patient age must be a whole number between 18 and 120.");
  }
  const result = await submitVoiceAssessment(file, patientAge, recordingTask);
  return fusionItem("VOICE", {
    type: "Voice",
    prediction: result.prediction,
    features: result.extracted_features,
    transcript: result.transcript,
    voiceDetails: {
      filename: result.filename,
      recording_task: result.recording_task,
      patient_age: result.patient_age,
      extraction_quality: result.extraction_quality,
      quality_notes: result.quality_notes,
      extraction_disclaimer: result.extraction_disclaimer,
    },
  });
}

/** Extract the biomedical record with Gemini, then score the extracted values as-is. */
export async function runBiomedicalUpload(file: File): Promise<HistoryItem> {
  const extraction = await extractBiomedicalRecord(file);
  if (extraction.extracted_count === 0) {
    throw new Error("No biomedical values could be read from this record. Use the Biomedical Assessment page to enter them manually.");
  }
  const prediction = await submitGeneralAssessment(extraction.features);
  return fusionItem("BIO", { type: "General", prediction, biomarkers: extraction.features });
}

export async function runEegUpload(files: File[], onProgress: (job: EegJob) => void): Promise<HistoryItem> {
  if (!files.some((file) => file.name.toLowerCase().endsWith(".set"))) {
    throw new Error("An EEGLAB .set file is required (with its .fdt companion if it has one).");
  }
  const created = await submitEegAssessment(files);
  onProgress(created);
  const settled = await pollEegJob(created.job_id, onProgress);
  if (settled.status !== "completed" || !settled.report) {
    throw new Error(settled.error?.message ?? "The EEG assessment could not be completed.");
  }
  return fusionItem("EEG", { type: "EEG", eegReport: settled.report });
}

export async function runNeuroimagingUpload(
  condition: UploadCondition,
  modeId: string,
  files: Record<string, File>,
): Promise<HistoryItem> {
  const neuroimaging = await runImagingUpload(condition, modeId, files);
  return fusionItem(`IMG-${condition}`, { type: "Neuroimaging", neuroimaging });
}
