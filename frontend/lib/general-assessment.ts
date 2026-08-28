import type {
  BiomedicalFeatures,
  BiomedicalInputKey,
  Prediction,
} from "@/lib/types";

type SelectOption = { label: string; value: string };

export type BiomarkerDetail = {
  label: string;
  unit: string;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  options?: readonly SelectOption[];
};

const binaryOptions = [
  { label: "No", value: "0" },
  { label: "Yes", value: "1" },
] as const;

export const biomarkerDetails: Record<BiomedicalInputKey, BiomarkerDetail> = {
  age: { label: "Age", unit: "years", description: "Age at assessment.", min: 18, max: 120, step: 1 },
  sex: {
    label: "Sex",
    unit: "",
    description: "Biological sex category accepted by the fitted encoder.",
    options: [
      { label: "Female", value: "Female" },
      { label: "Male", value: "Male" },
    ],
  },
  education_years: { label: "Education", unit: "years", description: "Completed years of formal education.", min: 0, max: 40 },
  bmi: { label: "Body mass index", unit: "kg/m²", description: "Body mass index at assessment.", min: 10, max: 80, step: 0.1 },
  family_history_pd: { label: "Family history of PD", unit: "", description: "Whether Parkinson's disease is reported in the family history.", options: binaryOptions },
  systolic_bp: { label: "Systolic blood pressure", unit: "mmHg", description: "Reported systolic blood pressure.", min: 50, max: 300 },
  diastolic_bp: { label: "Diastolic blood pressure", unit: "mmHg", description: "Reported diastolic blood pressure.", min: 30, max: 200 },
  cognitive_screen_score_0_30: { label: "Cognitive screen", unit: "/ 30", description: "Cognitive screening score on the model's 0–30 scale.", min: 0, max: 30 },
  rem_sleep_score: { label: "REM sleep score", unit: "score", description: "REM-sleep behavior score using the same instrument as the source data.", min: 0, max: 20 },
  updrs_part_i: { label: "UPDRS Part I", unit: "score", description: "Non-motor experiences of daily living, 0–52.", min: 0, max: 52 },
  updrs_part_ii: { label: "UPDRS Part II", unit: "score", description: "Motor experiences of daily living, 0–52.", min: 0, max: 52 },
  updrs_part_iii: { label: "UPDRS Part III", unit: "score", description: "Motor examination score, 0–132.", min: 0, max: 132 },
  updrs_part_iv: { label: "UPDRS Part IV", unit: "score", description: "Motor complications score, 0–24.", min: 0, max: 24 },
  schwab_england_adl: { label: "Schwab & England ADL", unit: "%", description: "Reported activities-of-daily-living independence, 0–100%.", min: 0, max: 100 },
  apoe_e4_count: {
    label: "APOE ε4 allele count",
    unit: "alleles",
    description: "Number of APOE ε4 alleles (0, 1, or 2).",
    options: [
      { label: "0 alleles", value: "0" },
      { label: "1 allele", value: "1" },
      { label: "2 alleles", value: "2" },
    ],
  },
  gba_variant_carrier: { label: "GBA variant carrier", unit: "", description: "Whether a GBA variant is reported.", options: binaryOptions },
  amyloid_beta_42_40_ratio: { label: "Amyloid beta 42/40 ratio", unit: "ratio", description: "Reported Aβ42/Aβ40 ratio; assay methods are not interchangeable.", min: 0, max: 1, step: 0.001 },
  t_tau_pg_ml: { label: "Total tau", unit: "pg/mL", description: "Reported total tau concentration.", min: 0 },
  p_tau181_pg_ml: { label: "Phosphorylated tau 181", unit: "pg/mL", description: "Reported p-tau181 concentration.", min: 0 },
  nfl_pg_ml: { label: "Neurofilament light (NfL)", unit: "pg/mL", description: "Reported neurofilament light concentration.", min: 0 },
  gfap_pg_ml: { label: "GFAP", unit: "pg/mL", description: "Reported glial fibrillary acidic protein concentration.", min: 0 },
  alpha_synuclein_pg_ml: { label: "Alpha-synuclein", unit: "pg/mL", description: "Reported quantitative alpha-synuclein concentration.", min: 0 },
  gdf15_pg_ml: { label: "GDF15", unit: "pg/mL", description: "Reported growth differentiation factor 15 concentration.", min: 0 },
  crp40_copy_number: { label: "CRP40 copy number", unit: "copies", description: "Reported CRP40 copy-number measurement.", min: 0 },
};

export const biomarkerGroups: {
  title: string;
  description: string;
  keys: BiomedicalInputKey[];
}[] = [
  {
    title: "Participant and vital context",
    description: "Demographics, family history, body composition, and blood pressure.",
    keys: ["age", "sex", "education_years", "bmi", "family_history_pd", "systolic_bp", "diastolic_bp"],
  },
  {
    title: "Cognitive, sleep, and motor scales",
    description: "Use the stated clinical instruments and enter their reported totals.",
    keys: ["cognitive_screen_score_0_30", "rem_sleep_score", "updrs_part_i", "updrs_part_ii", "updrs_part_iii", "updrs_part_iv", "schwab_england_adl"],
  },
  {
    title: "Genetic context",
    description: "Two encoded genetic risk indicators; leave blank when not tested.",
    keys: ["apoe_e4_count", "gba_variant_carrier"],
  },
  {
    title: "Neurological biomarkers",
    description: "Enter the laboratory-reported value and exact unit; assay methods can differ.",
    keys: ["amyloid_beta_42_40_ratio", "t_tau_pg_ml", "p_tau181_pg_ml", "nfl_pg_ml", "gfap_pg_ml", "alpha_synuclein_pg_ml", "gdf15_pg_ml", "crp40_copy_number"],
  },
];

function diseaseName(value: string) {
  return ({ AD: "Alzheimer-type", PD: "Parkinson-type", MS: "multiple-sclerosis-type", Healthy: "healthy-reference" } as Record<string, string>)[value] ?? value;
}

function riskBandExplanation(level: string) {
  if (level === "high") return "The independent risk model selected High. Arrange timely clinical review, while remembering that the result cannot establish a diagnosis.";
  if (level === "medium") return "The independent risk model selected Medium. Review the mixed pattern, input completeness, and test context with a qualified clinician.";
  return "The independent risk model selected Low. This is reassuring only inside this research model and cannot rule out neurological disease.";
}

export function buildGeneralInterpretation(prediction: Prediction) {
  const sortedProbabilities = Object.entries(prediction.probabilities).sort(([, left], [, right]) => right - left);
  const runnerUp = sortedProbabilities[1];
  const margin = runnerUp ? Math.max(0, prediction.confidence_score - runnerUp[1]) : prediction.confidence_score;
  const riskScore = Math.round(prediction.risk_score * 100);
  const confidence = Math.round(prediction.confidence_score * 100);
  const favorable = prediction.predicted_class === "Healthy" && prediction.risk_level === "low";
  const selectedName = diseaseName(prediction.predicted_class);

  const resultMeaning = `The disease pipeline matched the 24-input record most closely to its ${selectedName} class (${confidence}% model confidence). This describes similarity to a learned synthetic-cohort pattern—not a diagnosis, prevalence estimate, or laboratory interpretation.`;
  const formula = `A separate Low/Medium/High pipeline produced the ${prediction.risk_level} band and ${riskScore}/100 score. The saved formula weights Low at 0, Medium at 50, and High at 100, then averages those class probabilities. Disease confidence and overall risk are therefore different outputs.`;
  const separation = runnerUp
    ? `The next-closest disease class was ${diseaseName(runnerUp[0])} at ${Math.round(runnerUp[1] * 100)}%, a ${Math.round(margin * 100)}-point separation. ${margin < 0.15 ? "The narrow margin indicates a mixed disease-pattern output, so use extra caution." : "This separation is model certainty, not clinical certainty."}`
    : "No runner-up disease class was returned, so class separation could not be assessed.";

  return {
    lowerRiskHealthy: favorable,
    headline: `${selectedName} pattern · ${prediction.risk_level} risk`,
    resultMeaning,
    formula,
    separation,
    bandExplanation: riskBandExplanation(prediction.risk_level),
    suggestions: Array.from(new Set(prediction.recommendations)),
  };
}

export function formatBiomarkerValue(key: BiomedicalInputKey, features: BiomedicalFeatures) {
  const value = features[key];
  if (value === null) return "Not supplied";
  if (typeof value === "string") return value;
  if (key === "family_history_pd" || key === "gba_variant_carrier") {
    return value === 1 ? "Yes" : "No";
  }
  if (key === "apoe_e4_count") {
    return `${value} ${value === 1 ? "allele" : "alleles"}`;
  }
  const formatted = Number.isInteger(value)
    ? value.toLocaleString()
    : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  const unit = biomarkerDetails[key].unit;
  return unit ? `${formatted} ${unit}` : formatted;
}
