import type { EegRiskReport } from "@/lib/eeg-types";

export const featureKeys = [
  "mfcc_1_mean",
  "mfcc_2_mean",
  "mfcc_3_mean",
  "pitch_mean",
  "pitch_std",
  "jitter",
  "shimmer",
  "hnr",
  "speech_rate",
  "pause_count",
  "mean_pause_duration",
  "mean_energy",
  "spectral_centroid_mean",
  "zero_crossing_rate_mean",
] as const;

export type FeatureKey = (typeof featureKeys)[number];
export type SpeechFeatures = Record<FeatureKey, number>;

export type Prediction = {
  predicted_class: string;
  confidence_score: number;
  risk_score: number;
  risk_level: string;
  probabilities: Record<string, number>;
  observed_issues: string[];
  recommendations: string[];
  disclaimer: string;
};

export type VoiceAssessment = {
  filename: string;
  recording_task: string;
  patient_age: number;
  extraction_quality: "usable" | "limited";
  quality_notes: string[];
  transcript: string;
  extracted_features: SpeechFeatures;
  prediction: Prediction;
  extraction_disclaimer: string;
};

export type AssessmentType = "Voice" | "General" | "EEG";

export type HistoryItem = {
  id: string;
  createdAt: string;
  type: AssessmentType;
  /**
   * Speech-model output. Optional because EEG rows carry `eegReport` instead —
   * the EEG model emits three independent risk scores, not one class plus a
   * softmax, so it cannot be squeezed into this shape without misrepresenting it.
   * Anything reading this field must handle it being absent.
   */
  prediction?: Prediction;
  eegReport?: EegRiskReport;
  features?: SpeechFeatures;
  transcript?: string;
};
