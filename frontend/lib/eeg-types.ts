export const riskConditions = ["AD", "PD", "MS"] as const;
export type RiskCondition = (typeof riskConditions)[number];
export type RiskBand = "Low" | "Medium" | "High";

export type ConditionRisk = {
  risk_score: number;
  risk_band: RiskBand;
  label: string;
  epoch_score_std: number;
  epoch_score_range: number[];
  confound_severity: string;
};

export type RiskAssessment = {
  conditions: Record<string, ConditionRisk>;
  highest_risk_condition: string;
  /** Always true. Independent sigmoids — the scores do not sum to 1. */
  scores_are_independent: true;
  interpretation: string;
  risk_bands: Record<string, number>;
};

export type IcaRejection = {
  component: number;
  criteria: string[];
  kurtosis?: number | null;
  frontal_corr?: number | null;
  hf_power_ratio?: number | null;
};

export type EegSignalQuality = {
  epochs_used: number;
  total_epochs_generated: number;
  clean_epoch_ratio: number;
  grade: string;
  ica_components_removed: number;
  ica_rejections: IcaRejection[];
  channels: number;
  sampling_rate_hz: number;
  source_kind: string;
  warnings: string[];
};

export type EegEmbeddingSummary = {
  dim: number;
  l2_norm: number;
  availability_flag: 0 | 1;
  consistency: number;
  cosine_to_class_centroids: Record<string, number>;
  nearest_centroid: string | null;
  vector_url: string | null;
};

export type EegExplainability = {
  scalp_region_importance: Record<string, number>;
  band_importance: Record<string, number>;
  method: string;
};

export type ConfoundDisclosure = {
  age_probe_mae_years: number | null;
  age_probe_improvement_over_baseline: number | null;
  site_probe_balanced_accuracy: number | null;
  risk_score_age_correlation: Record<string, number>;
  severity_by_condition: Record<string, string>;
  statement: string;
  /** Drives the render-blocking banner on the results page. */
  has_critical: boolean;
};

export type FourClassPrediction = {
  predicted_class: string;
  class_probabilities: Record<string, number>;
  note: string;
};

export type EegRiskReport = {
  subject_id: string;
  source: "cohort" | "upload";
  generated_at: string;
  dataset: Record<string, string>;
  risk_scores: Record<string, number>;
  risk_assessment: RiskAssessment;
  optional_four_class_prediction: FourClassPrediction | null;
  signal_quality: EegSignalQuality;
  band_power_profile: Record<string, number>;
  embedding: EegEmbeddingSummary;
  explainability: EegExplainability;
  confound_disclosure: ConfoundDisclosure;
  model_summary: Record<string, string | number | null>;
  clinical_disclaimer: string;
};

export type CohortSubject = {
  subject_id: string;
  true_class: string;
  site: string;
  source_kind: string;
  signal_quality: string;
  epochs_used: number;
  age: number | null;
  highest_risk_condition: string;
  highest_risk_score: number;
  risk_scores: Record<string, number>;
  confound_severity: string;
};

export type CohortPage = {
  total: number;
  offset: number;
  limit: number;
  subjects: CohortSubject[];
  available_filters: Record<string, string[]>;
};

export type ProjectionPoint = {
  subject_id: string;
  x: number;
  y: number;
  true_class: string;
  site: string;
};

export type CohortProjection = {
  method: string;
  points: ProjectionPoint[];
  explained_variance: number[] | null;
  /**
   * Nearest-neighbour class agreement measured in the full 256-D space, keyed by
   * "overall" plus one entry per class. The scatter shows two components carrying a
   * fraction of the variance, so it can look unstructured while the geometry behind
   * it separates cleanly — this is the number that settles which.
   */
  neighbourhood_agreement: Record<string, number>;
  note: string;
};

export type EegModelCard = {
  run_id: string;
  generated_at: string | null;
  architecture: string;
  input_representation: string;
  input_shape: number[];
  embedding_dim: number;
  risk_conditions: string[];
  cohort: Record<string, unknown>;
  performance: Record<string, unknown>;
  confound_disclosure: ConfoundDisclosure;
  intended_use: Record<string, unknown>;
  inference_available: boolean;
};

export type GroupBandSummary = {
  n: number;
  medians: Record<string, number>;
  q1: Record<string, number>;
  q3: Record<string, number>;
};

export type ConditionBandProfile = {
  condition: string;
  n: number;
  medians: Record<string, number>;
  auc_vs_hc: Record<string, number>;
  direction_vs_hc: Record<string, string>;
  separating_bands: string[];
  opposing_bands: string[];
  /**
   * False means this cohort shows no band-power pattern for the condition. The UI
   * must then decline to point at one — on this cohort MS reads as a younger group,
   * not an affected one, and dressing that up would be a fabricated finding.
   */
  has_signature: boolean;
  note: string;
};

/** Cohort band statistics. Descriptive context, never an attribution of the score. */
export type BandReference = {
  generated_at: string;
  bands: string[];
  separation_margin: number;
  canonical_axis: string;
  slowing_direction: Record<string, string>;
  healthy: GroupBandSummary;
  conditions: Record<string, ConditionBandProfile>;
  method: string;
};

export type EegJobStatus =
  | "queued"
  | "validating"
  | "preprocessing"
  | "inference"
  | "completed"
  | "failed";

export type EegJob = {
  job_id: string;
  status: EegJobStatus;
  progress: number;
  stage_label: string;
  filename: string | null;
  created_at: string;
  updated_at: string;
  report: EegRiskReport | null;
  error: EegJobError | null;
};

export type EegJobError = {
  code: string;
  message: string;
  details: EegRejectionDetails | Record<string, unknown>;
};

/** Why a recording was rejected, in terms the uploader can act on. */
export type EegRejectionDetails = {
  reason?: string;
  epochs_generated?: number;
  epochs_surviving?: number;
  epochs_required?: number;
  threshold_uv?: number;
  median_epoch_peak_to_peak_uv?: number;
  max_epoch_peak_to_peak_uv?: number;
  channels_over_threshold?: number;
  total_channels?: number;
  worst_channels?: { channel: string; peak_to_peak_uv: number }[];
};

/** Severity strings that must block rendering behind a persistent banner. */
export function isCriticalSeverity(severity: string): boolean {
  return severity.toUpperCase().includes("CRITICAL");
}

export function isFlaggedSeverity(severity: string): boolean {
  const value = severity.toUpperCase();
  return value.includes("CRITICAL") || value.includes("HIGH") || value.includes("MODERATE");
}
