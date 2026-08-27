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

export const biomedicalNumericKeys = [
  "age",
  "education_years",
  "bmi",
  "family_history_pd",
  "systolic_bp",
  "diastolic_bp",
  "cognitive_screen_score_0_30",
  "rem_sleep_score",
  "updrs_part_i",
  "updrs_part_ii",
  "updrs_part_iii",
  "updrs_part_iv",
  "schwab_england_adl",
  "apoe_e4_count",
  "gba_variant_carrier",
  "amyloid_beta_42_40_ratio",
  "t_tau_pg_ml",
  "p_tau181_pg_ml",
  "nfl_pg_ml",
  "gfap_pg_ml",
  "alpha_synuclein_pg_ml",
  "gdf15_pg_ml",
  "crp40_copy_number",
] as const;

export const biomedicalInputKeys = [
  "age",
  "sex",
  "education_years",
  "bmi",
  "family_history_pd",
  "systolic_bp",
  "diastolic_bp",
  "cognitive_screen_score_0_30",
  "rem_sleep_score",
  "updrs_part_i",
  "updrs_part_ii",
  "updrs_part_iii",
  "updrs_part_iv",
  "schwab_england_adl",
  "apoe_e4_count",
  "gba_variant_carrier",
  "amyloid_beta_42_40_ratio",
  "t_tau_pg_ml",
  "p_tau181_pg_ml",
  "nfl_pg_ml",
  "gfap_pg_ml",
  "alpha_synuclein_pg_ml",
  "gdf15_pg_ml",
  "crp40_copy_number",
] as const;

export type BiomedicalNumericKey = (typeof biomedicalNumericKeys)[number];
export type BiomedicalInputKey = (typeof biomedicalInputKeys)[number];
export type BiomedicalFeatures = {
  sex: "Female" | "Male" | null;
} & Record<BiomedicalNumericKey, number | null>;

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
  biomarkers?: BiomedicalFeatures;
  transcript?: string;
};
