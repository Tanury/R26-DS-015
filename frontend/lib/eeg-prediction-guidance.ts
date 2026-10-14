import type { EegRiskReport } from "@/lib/eeg-types";

export const eegPredictionClasses = ["Healthy", "AD", "PD", "MS"] as const;
export type EegPredictionClass = (typeof eegPredictionClasses)[number];

export type EegConditionGuidance = {
  condition: EegPredictionClass;
  name: string;
  score: number | null;
  points: number | null;
  riskScore: number | null;
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
  scoreLabel: string;
  evidence: string[];
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

function validScore(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
}

// Allocate 1,000 tenths of a percent using largest remainders so the displayed
// distribution totals exactly 100.0%, including rounded API probabilities.
function fourClassPoints(probabilities: Partial<Record<EegPredictionClass, number>>) {
  const total = eegPredictionClasses.reduce((sum, condition) => sum + probabilities[condition]!, 0);
  const shares = eegPredictionClasses.map((condition) => {
    const exact = probabilities[condition]! / total * 1000;
    return { condition, units: Math.floor(exact), remainder: exact - Math.floor(exact) };
  });
  const remaining = 1000 - shares.reduce((sum, share) => sum + share.units, 0);
  const byRemainder = [...shares].sort((a, b) => b.remainder - a.remainder);
  for (let index = 0; index < remaining; index++) byRemainder[index].units++;
  return Object.fromEntries(shares.map(({ condition, units }) => [condition, units / 10])) as Record<EegPredictionClass, number>;
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
  const probabilities: Partial<Record<EegPredictionClass, number>> = {};
  for (const [label, value] of Object.entries(report.optional_four_class_prediction?.class_probabilities ?? {})) {
    const condition = normalizePredictionClass(label);
    const score = validScore(value);
    if (condition && score !== null) probabilities[condition] = score;
  }
  // Never mix the independent sigmoid heads with the auxiliary softmax distribution.
  const hasFourClass = eegPredictionClasses.every((condition) => probabilities[condition] !== undefined)
    && Math.abs(Object.values(probabilities).reduce((sum, value) => sum + value, 0) - 1) < 0.02;
  const riskScore = (condition: EegPredictionClass) => condition === "Healthy" ? null : validScore(
    report.risk_assessment.conditions[condition]?.risk_score
      ?? report.risk_scores[`${condition.toLowerCase()}_risk_score`]
      ?? report.risk_scores[condition],
  );
  const scores = eegPredictionClasses.map((condition) => ({
    condition, score: hasFourClass ? probabilities[condition]! : riskScore(condition),
  }));
  const ranked = scores.filter((entry): entry is {condition: EegPredictionClass; score: number} => entry.score !== null)
    .sort((left, right) => right.score - left.score);
  const selected = ranked[0];
  const runnerUp = ranked[1];
  const tied = !!selected && !!runnerUp && Math.abs(selected.score - runnerUp.score) < 1e-8;
  const scoreLabel = hasFourClass ? "Four-class model score" : "Independent risk score";
  const balancedPoints = hasFourClass ? fourClassPoints(probabilities) : null;
  const pointsFor = (condition: EegPredictionClass, score: number) =>
    balancedPoints?.[condition] ?? Math.round(score * 1000) / 10;
  const conditions = eegPredictionClasses.map((condition) => {
    const score = scores.find((entry) => entry.condition === condition)!.score;
    const points = score === null ? null : pointsFor(condition, score);
    const isPrimary = !tied && condition === selected?.condition;
    const explanation = score === null
      ? "No measured score is available for this class. It cannot be ranked or ruled out."
      : isPrimary
        ? `Highest ${scoreLabel.toLowerCase()} at ${points!.toFixed(1)}%${runnerUp ? `, ${(points! - pointsFor(runnerUp.condition, runnerUp.score)).toFixed(1)} percentage points above ${runnerUp.condition}` : ""}.`
        : selected && Math.abs(score - selected.score) < 1e-8
          ? `Joint highest measured score; displayed as ${points!.toFixed(1)}%. The model does not prefer one tied class.`
          : `${points!.toFixed(1)}% is ${((selected ? pointsFor(selected.condition, selected.score) : points!) - points!).toFixed(1)} percentage points below ${selected?.condition}. A lower score does not rule this condition out.`;
    return {
      condition,
      name: conditionNames[condition],
      score,
      points,
      riskScore: riskScore(condition),
      isPrimary,
      explanation,
    } satisfies EegConditionGuidance;
  });
  const top = conditions.find((condition) => condition.condition === selected?.condition) ?? conditions[0];

  const headline = !selected ? "EEG result unavailable" : tied ? "No single preferred EEG pattern"
    : hasFourClass ? `${top.condition} EEG pattern selected` : `${top.condition} has the highest disease-risk score`;
  const summary = !selected ? "No valid model scores were saved for this recording. Run a new assessment."
    : tied ? "Two or more classes share the highest measured score. Review the alternatives and signal quality."
      : `${top.name} has the highest ${scoreLabel.toLowerCase()} (${top.points}%). This is a research pattern match, not a confirmed diagnosis.`;

  const quality = report.signal_quality;
  const qualityName = quality.grade || "Unknown";
  const signalQualitySummary = `${qualityName} signal quality: ${Math.round(quality.clean_epoch_ratio * 100)}% of ${quality.total_epochs_generated} generated epochs passed quality checks; ${quality.epochs_used} were used for scoring after sampling. ${quality.ica_components_removed} ICA component${quality.ica_components_removed === 1 ? " was" : "s were"} removed.`;
  const recommendations = selected && !tied ? recommendationsFor(top.condition) : ["Review this inconclusive result with a qualified healthcare professional alongside symptoms and signal quality."];
  if (qualityName.toLowerCase() !== "good" || quality.warnings.length > 0) {
    recommendations.push("Review the signal-quality warnings and consider repeating the EEG under standardized recording conditions before relying on this result.");
  }

  const cautions = [
    hasFourClass
      ? "Four-class scores come from the auxiliary model head. Displayed percentages are normalized and rounded together to total exactly 100%. The separate AD, PD and MS risk scores are independent and may be elevated together. Neither output is a calibrated clinical probability."
      : "The four-class output is unavailable. Only measured AD, PD and MS risk scores are ranked; a healthy-control score cannot be inferred by subtracting them from 100%.",
    "This page explains an existing model output; it does not perform a new prediction or provide a medical diagnosis.",
  ];
  if (runnerUp && selected && selected.score - runnerUp.score < 0.05) {
    cautions.push("The two leading scores are less than 5 percentage points apart. Their numerical separation is small; this display flag is not a validated clinical threshold.");
  }
  const elevated = conditions.filter((entry) => entry.riskScore !== null && entry.riskScore > (report.risk_assessment.risk_bands.low_max ?? 0.39));
  if (elevated.length > 1 || (top.condition === "Healthy" && elevated.length)) {
    cautions.push(`Independent risk heads are elevated for ${elevated.map((entry) => entry.condition).join(", ")}. The four-class selection must not hide these findings.`);
  }
  if (!hasFourClass && selected && !elevated.length) {
    cautions.push("All available disease-risk scores are in the low band. Having the highest score does not mean a disease pattern is established.");
  }
  const evidence: string[] = [hasFourClass
    ? "The selected pattern is the largest measured output of the four-class head. The comparisons below use that same output for every class."
    : "The available independent disease-risk heads are ranked by their measured scores. Healthy control is not scored by these heads."];
  const target = report.explainability.target_condition;
  const regions = Object.entries(report.explainability.scalp_region_importance)
    .filter(([, value]) => Number.isFinite(value) && value > 0).sort((a, b) => b[1] - a[1]);
  if (regions.length && target && report.explainability.target_output === "independent_risk") {
    evidence.push(`Occlusion evidence targets the ${target} independent risk head: zeroing ${regions[0][0].replaceAll("_", " ")} reduced its score by ${(regions[0][1] * 100).toFixed(1)} percentage points. This is input sensitivity, not proof of disease${hasFourClass ? " or an attribution of the four-class selection" : ""}.`);
  } else if (regions.length) {
    evidence.push(`Stored occlusion reports a drop of ${(regions[0][1] * 100).toFixed(1)} percentage points when ${regions[0][0].replaceAll("_", " ")} is zeroed. This report does not identify a compatible output head being explained, so this cannot be attributed to the selected score.`);
  } else {
    evidence.push("No positive regional occlusion evidence is available in this report. The score ranking alone does not tell us which signal features caused the selection.");
  }
  const nearest = normalizePredictionClass(report.embedding.nearest_centroid);
  if (nearest && report.embedding.availability_flag === 1) {
    evidence.push(`The learned EEG representation is closest to the ${nearest} class centroid${selected && nearest !== top.condition ? ", which disagrees with the selected score" : ""}. Centroid proximity is descriptive and does not confirm a diagnosis.`);
  }
  if (report.confound_disclosure.statement) cautions.push(report.confound_disclosure.statement);
  const severities = Object.entries(report.confound_disclosure.severity_by_condition ?? {})
    .filter(([, severity]) => /critical|high|moderate/i.test(severity));
  if (severities.length) cautions.push(`Model confounds: ${severities.map(([condition, severity]) => `${condition}: ${severity}`).join("; ")}. Age or recording site may influence these scores.`);
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
    scoreLabel,
    evidence,
  };
}
