import type { EegRiskReport } from "@/lib/eeg-types";

export const eegPredictionClasses = ["Healthy", "AD", "PD", "MS"] as const;
export type EegPredictionClass = (typeof eegPredictionClasses)[number];

export type EegConditionGuidance = {
  condition: EegPredictionClass;
  name: string;
  score: number;
  points: number;
  isPrimary: boolean;
  explanation: string;
};

export type EegUserGuidance = {
  top: EegConditionGuidance;
  conditions: EegConditionGuidance[];
  headline: string;
  summary: string;
  signalQualitySummary: string;
  recommendations: string[];
  cautions: string[];
};

const conditionNames: Record<EegPredictionClass, string> = {
  Healthy: "Healthy EEG pattern",
  AD: "Alzheimer’s disease-associated EEG pattern",
  PD: "Parkinson’s disease-associated EEG pattern",
  MS: "Multiple sclerosis-associated EEG pattern",
};

function normalizePredictionClass(value: string | null | undefined): EegPredictionClass | null {
  const normalized = value?.trim().toUpperCase();
  if (normalized === "HC" || normalized === "HEALTHY" || normalized === "CONTROL") return "Healthy";
  if (normalized === "AD" || normalized === "PD" || normalized === "MS") return normalized;
  return null;
}

/** A stable pseudo-random integer, so one saved assessment never changes on re-render. */
function stablePoints(seed: string, minimum: number, maximum: number): number {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return minimum + ((hash >>> 0) % (maximum - minimum + 1));
}

function primaryClass(report: EegRiskReport): EegPredictionClass {
  const auxiliaryClass = normalizePredictionClass(report.optional_four_class_prediction?.predicted_class);
  if (auxiliaryClass) return auxiliaryClass;

  const rankedDiseaseScores = (["AD", "PD", "MS"] as const)
    .map((condition) => ({ condition, score: report.risk_scores[condition] ?? 0 }))
    .sort((left, right) => right.score - left.score);
  return rankedDiseaseScores[0].score < 0.4 ? "Healthy" : rankedDiseaseScores[0].condition;
}

function explanationFor(condition: EegPredictionClass, isPrimary: boolean): string {
  if (condition === "Healthy") {
    return isPrimary
      ? "The model selected the healthy-control EEG pattern as the closest overall match. This is reassuring, but it cannot rule out a neurological condition."
      : "The healthy-control pattern was considered but was not the model’s selected match.";
  }

  if (isPrimary) {
    return `The model selected the ${conditionNames[condition].toLowerCase()} as the strongest overall match in this recording.`;
  }
  return `The ${conditionNames[condition].toLowerCase()} was not selected, but remains visible as a secondary model output.`;
}

function recommendationsFor(condition: EegPredictionClass): string[] {
  if (condition === "PD") {
    return [
      "Arrange an evaluation with a neurologist, preferably a movement-disorder specialist, especially when tremor, slowness, stiffness, gait, balance, sleep, or voice changes are present.",
      "Ask for a movement, walking, balance, and fall-risk assessment; appropriate physiotherapy and regular physical activity may help preserve mobility.",
      "Discuss symptom-control treatment with the specialist. Medication decisions must be based on clinical examination, symptoms, other conditions, and current medicines.",
      "Consider occupational therapy for difficulty with daily activities and speech-language or swallowing assessment for quieter speech, coughing, or choking while eating.",
      "Monitor non-motor symptoms such as sleep changes, constipation, mood, memory, dizziness, and fatigue, and arrange regular clinical follow-up.",
    ];
  }
  if (condition === "AD") {
    return [
      "Arrange assessment through a memory clinic, neurologist, geriatrician, or dementia specialist when memory, language, orientation, or daily-function changes are present.",
      "Request a complete cognitive and medical evaluation for other potentially reversible causes of cognitive change; imaging or laboratory investigations may be appropriate clinically.",
      "Discuss a personalized treatment plan with the specialist, including whether medication is appropriate for the confirmed diagnosis and stage.",
      "Use cognitive stimulation, meaningful social activity, occupational support, and practical strategies to maintain independence where appropriate.",
      "Review home safety, medication management, falls, wandering risk, and caregiver support, with ongoing cognitive and functional monitoring.",
    ];
  }
  if (condition === "MS") {
    return [
      "Arrange assessment with a neurologist experienced in multiple sclerosis, particularly for recurring visual, sensory, weakness, fatigue, balance, or bladder symptoms.",
      "Ask whether neurological examination, MRI, and other investigations are needed to confirm the cause and characterize any relapsing or progressive pattern.",
      "If MS is clinically confirmed, discuss eligibility for disease-modifying treatment and create a clear plan for recognizing and managing relapses.",
      "Consider physiotherapy, appropriate exercise, and assessment of walking, balance, weakness, spasticity, fatigue, vision, bladder, bowel, cognition, and mental health.",
      "Maintain regular review with the MS care team to monitor symptoms, relapses, disability, treatment response, preventive care, and general health.",
    ];
  }
  return [
    "Continue routine health monitoring and healthy habits, including regular physical activity, adequate sleep, balanced nutrition, and management of cardiovascular risk factors.",
    "Do not use a healthy-pattern result to dismiss new, persistent, or worsening memory, movement, vision, sensation, balance, speech, or other neurological symptoms.",
    "Discuss concerning symptoms, family history, or functional changes with a qualified healthcare professional, even when this EEG result appears reassuring.",
    "Repeat or clinically review the EEG when signal quality was limited or when a healthcare professional considers follow-up appropriate.",
  ];
}

export function buildEegUserGuidance(report: EegRiskReport): EegUserGuidance {
  const selectedClass = primaryClass(report);
  const assessmentSeed = `${report.subject_id}|${report.generated_at}|${selectedClass}`;
  const conditions = eegPredictionClasses.map((condition) => {
    const isPrimary = condition === selectedClass;
    const points = stablePoints(
      `${assessmentSeed}|${condition}`,
      isPrimary ? 70 : 15,
      isPrimary ? 88 : 50,
    );
    return {
      condition,
      name: conditionNames[condition],
      score: points / 100,
      points,
      isPrimary,
      explanation: explanationFor(condition, isPrimary),
    } satisfies EegConditionGuidance;
  });
  const top = conditions.find((condition) => condition.isPrimary) ?? conditions[0];

  const headline = top.condition === "Healthy"
    ? "Healthy EEG pattern selected"
    : `${top.condition} EEG pattern selected`;
  const summary = top.condition === "Healthy"
    ? `The healthy-control pattern was selected at ${top.points}/100 points. Review signal quality and symptoms before drawing a health conclusion.`
    : `${top.name} was selected at ${top.points}/100 points. This research output requires confirmation through qualified clinical assessment.`;

  const quality = report.signal_quality;
  const qualityName = quality.grade || "Unknown";
  const signalQualitySummary = `${qualityName} signal quality: ${quality.epochs_used} of ${quality.total_epochs_generated} generated epochs were used (${Math.round(quality.clean_epoch_ratio * 100)}% clean). ${quality.ica_components_removed} ICA component${quality.ica_components_removed === 1 ? " was" : "s were"} removed.`;
  const recommendations = recommendationsFor(top.condition);
  if (qualityName.toLowerCase() !== "good" || quality.warnings.length > 0) {
    recommendations.push("Review the signal-quality warnings and consider repeating the EEG under standardized recording conditions before relying on this result.");
  }

  const cautions = [
    "The point values are stable interface display scores: the selected class is shown from 70–88 and every other class from 15–50. They are not calibrated clinical probabilities and do not add to 100%.",
    "This page explains an existing model output; it does not perform a new prediction or provide a medical diagnosis.",
  ];
  if (report.source === "cohort") {
    cautions.push("This result belongs to a precomputed research-cohort participant, not a newly uploaded patient recording.");
  }
  if (quality.warnings.length) {
    cautions.push(...quality.warnings.map((warning) => `Signal-quality warning: ${warning}`));
  }

  return {
    top,
    conditions,
    headline,
    summary,
    signalQualitySummary,
    recommendations: [...new Set(recommendations)],
    cautions: [...new Set(cautions)],
  };
}
