import type { RiskCondition } from "@/lib/eeg-types";
import type { AssessmentType, HistoryItem, Prediction } from "@/lib/types";

export const fusionConditions = ["AD", "PD", "MS"] as const satisfies readonly RiskCondition[];
export type FusionCondition = (typeof fusionConditions)[number];
export type FusionModality = "Voice" | "General" | "EEG";
export type FusionRiskLevel = "Low" | "Moderate" | "High";

export type FusionContribution = {
  modality: FusionModality;
  evidence: number;
  effectiveWeight: number;
  qualityLabel: string;
  note: string;
};

export type FusedCondition = {
  condition: FusionCondition;
  score: number;
  level: FusionRiskLevel;
  contributions: FusionContribution[];
};

export type FusionResult = {
  generatedAt: string;
  conditions: Record<FusionCondition, FusedCondition>;
  highestCondition: FusionCondition;
  highestScore: number;
  riskLevel: FusionRiskLevel;
  agreement: "Strong" | "Partial" | "Discordant";
  agreeingModalities: number;
  headline: string;
  explanation: string;
  suggestions: string[];
  cautions: string[];
  sourceIds: Record<FusionModality, string>;
};

const conditionAliases: Record<FusionCondition, string[]> = {
  AD: ["AD", "Alzheimer", "Alzheimer's disease", "Alzheimer’s disease"],
  PD: ["PD", "Parkinson", "Parkinson's disease", "Parkinson’s disease"],
  MS: ["MS", "Multiple sclerosis"],
};

const healthyAliases = ["Healthy", "HC", "Control", "Healthy Control"];

function clamp(value: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

function probabilityFor(probabilities: Record<string, number>, aliases: string[]): number {
  const normalized = Object.entries(probabilities).map(([key, value]) => [key.trim().toLowerCase(), value] as const);
  for (const alias of aliases) {
    const match = normalized.find(([key]) => key === alias.toLowerCase());
    if (match) return clamp(Number(match[1]));
  }
  return 0;
}

/**
 * Voice and biomedical models emit mutually exclusive four-class probabilities,
 * while EEG emits independent condition risks. Comparing a disease class only
 * with Healthy produces a transparent one-vs-Healthy evidence value that can be
 * combined with the matching EEG head without pretending the softmax is a set of
 * independent risks.
 */
function oneVsHealthyEvidence(prediction: Prediction, condition: FusionCondition): number {
  const disease = probabilityFor(prediction.probabilities, conditionAliases[condition]);
  const healthy = probabilityFor(prediction.probabilities, healthyAliases);
  const denominator = disease + healthy;
  return denominator > 0 ? disease / denominator : disease;
}

function riskLevel(score: number): FusionRiskLevel {
  if (score >= 0.75) return "High";
  if (score >= 0.4) return "Moderate";
  return "Low";
}

function voiceWeight(item: HistoryItem): { weight: number; label: string; note: string } {
  if (item.voiceDetails?.extraction_quality === "limited") {
    return {
      weight: 0.75,
      label: "Limited extraction quality",
      note: "Gemini marked the recording as limited, so its influence is reduced to 75%.",
    };
  }
  if (!item.voiceDetails) {
    return {
      weight: 0.9,
      label: "Quality metadata unavailable",
      note: "This older result has no saved extraction-quality metadata, so its influence is reduced to 90%.",
    };
  }
  return {
    weight: 1,
    label: "Usable extraction",
    note: "The recording passed the voice extraction quality gate.",
  };
}

function biomedicalWeight(item: HistoryItem): { weight: number; label: string; note: string } {
  if (!item.biomarkers) {
    return {
      weight: 0.75,
      label: "Input completeness unavailable",
      note: "This older result has no saved input record, so its influence is reduced to 75%.",
    };
  }
  const values = Object.values(item.biomarkers);
  const supplied = values.filter((value) => value !== null && value !== undefined).length;
  const completeness = values.length ? supplied / values.length : 0;
  const weight = Math.max(0.5, completeness);
  const missing = values.length - supplied;
  return {
    weight,
    label: missing ? `${supplied}/${values.length} inputs supplied` : "All inputs supplied",
    note: missing
      ? `${missing} biomedical input${missing === 1 ? " was" : "s were"} imputed; influence is scaled to input completeness with a 50% floor.`
      : "All saved biomedical inputs were supplied.",
  };
}

function eegQualityWeight(grade: string): number {
  const normalized = grade.trim().toLowerCase();
  if (normalized === "good") return 1;
  if (normalized === "moderate") return 0.8;
  if (normalized === "poor") return 0.6;
  return 0.75;
}

function confoundWeight(severity: string): number {
  const normalized = severity.toUpperCase();
  if (normalized.includes("CRITICAL")) return 0.25;
  if (normalized.includes("HIGH")) return 0.5;
  if (normalized.includes("MODERATE")) return 0.75;
  return 1;
}

function topCondition(scores: Record<FusionCondition, number>): FusionCondition {
  return [...fusionConditions].sort((left, right) => scores[right] - scores[left])[0];
}

export function latestFusionAssessments(items: HistoryItem[]): Partial<Record<FusionModality, HistoryItem>> {
  const latest: Partial<Record<FusionModality, HistoryItem>> = {};
  for (const item of items) {
    if (!latest[item.type]) latest[item.type] = item;
  }
  return latest;
}

export function buildFusionResult(selected: Record<FusionModality, HistoryItem>): FusionResult {
  if (!selected.Voice.prediction || !selected.General.prediction || !selected.EEG.eegReport) {
    throw new Error("Voice, biomedical, and EEG model outputs are all required for fusion.");
  }

  const voiceQuality = voiceWeight(selected.Voice);
  const biomedicalQuality = biomedicalWeight(selected.General);
  const eegReport = selected.EEG.eegReport;
  const eegBaseWeight = eegQualityWeight(eegReport.signal_quality.grade);
  const modalityScores = {
    Voice: Object.fromEntries(fusionConditions.map((condition) => [
      condition,
      oneVsHealthyEvidence(selected.Voice.prediction!, condition),
    ])) as Record<FusionCondition, number>,
    General: Object.fromEntries(fusionConditions.map((condition) => [
      condition,
      oneVsHealthyEvidence(selected.General.prediction!, condition),
    ])) as Record<FusionCondition, number>,
    EEG: Object.fromEntries(fusionConditions.map((condition) => [
      condition,
      clamp(eegReport.risk_assessment.conditions[condition]?.risk_score ?? eegReport.risk_scores[condition] ?? 0),
    ])) as Record<FusionCondition, number>,
  };

  const conditions = Object.fromEntries(fusionConditions.map((condition) => {
    const severity = eegReport.risk_assessment.conditions[condition]?.confound_severity
      ?? eegReport.confound_disclosure.severity_by_condition[condition]
      ?? "None";
    const eegWeight = eegBaseWeight * confoundWeight(severity);
    const contributions: FusionContribution[] = [
      {
        modality: "Voice",
        evidence: modalityScores.Voice[condition],
        effectiveWeight: voiceQuality.weight,
        qualityLabel: voiceQuality.label,
        note: voiceQuality.note,
      },
      {
        modality: "General",
        evidence: modalityScores.General[condition],
        effectiveWeight: biomedicalQuality.weight,
        qualityLabel: biomedicalQuality.label,
        note: biomedicalQuality.note,
      },
      {
        modality: "EEG",
        evidence: modalityScores.EEG[condition],
        effectiveWeight: eegWeight,
        qualityLabel: `${eegReport.signal_quality.grade} signal · ${severity || "no"} confound`,
        note: `EEG influence combines the ${eegReport.signal_quality.grade} signal-quality factor with the documented ${condition} confound factor.`,
      },
    ];
    const denominator = contributions.reduce((sum, contribution) => sum + contribution.effectiveWeight, 0);
    const score = denominator
      ? contributions.reduce((sum, contribution) => sum + contribution.evidence * contribution.effectiveWeight, 0) / denominator
      : 0;
    const value: FusedCondition = {
      condition,
      score: clamp(score),
      level: riskLevel(score),
      contributions,
    };
    return [condition, value];
  })) as Record<FusionCondition, FusedCondition>;

  const fusedScores = Object.fromEntries(fusionConditions.map((condition) => [condition, conditions[condition].score])) as Record<FusionCondition, number>;
  const highestCondition = topCondition(fusedScores);
  const highestScore = fusedScores[highestCondition];
  const modalityLeaders = (Object.keys(modalityScores) as FusionModality[]).map((modality) => topCondition(modalityScores[modality]));
  const agreeingModalities = modalityLeaders.filter((condition) => condition === highestCondition).length;
  const agreement = agreeingModalities === 3 ? "Strong" : agreeingModalities === 2 ? "Partial" : "Discordant";
  const ranked = [...fusionConditions].sort((left, right) => fusedScores[right] - fusedScores[left]);
  const margin = fusedScores[ranked[0]] - fusedScores[ranked[1]];
  const severity = eegReport.risk_assessment.conditions[highestCondition]?.confound_severity
    ?? eegReport.confound_disclosure.severity_by_condition[highestCondition]
    ?? "None";
  const cautions = [
    "The selected assessments are not linked by a shared patient identifier. Verify that all three belong to the same participant and a comparable time period.",
    "This is deterministic late fusion of separately trained models, not a jointly trained or clinically calibrated multimodal model.",
  ];
  if (agreement !== "Strong") {
    cautions.push(`The component leaders are ${modalityLeaders.join(", ")}; the modalities do not fully agree.`);
  }
  if (/CRITICAL|HIGH|MODERATE/i.test(severity)) {
    cautions.push(`The ${highestCondition} EEG contribution carries a documented ${severity} confound and was down-weighted.`);
  }
  if (selected.Voice.voiceDetails?.extraction_quality === "limited") {
    cautions.push("Voice extraction quality was limited, so the voice contribution was down-weighted.");
  }

  const level = riskLevel(highestScore);
  const headline = level === "Low"
    ? "No elevated multimodal pattern"
    : `${highestCondition} multimodal risk pattern`;
  const explanation = level === "Low"
    ? `All combined disease scores remain in the Low band. ${highestCondition} is the highest at ${Math.round(highestScore * 100)}/100, but this does not rule out disease.`
    : `${highestCondition} has the highest quality-adjusted fused score at ${Math.round(highestScore * 100)}/100. ${agreeingModalities} of 3 component models rank ${highestCondition} first, with a ${Math.round(margin * 100)}-point lead over the next fused condition.`;
  const suggestions = level === "High"
    ? [
        "Arrange timely review with a qualified neurologist or appropriate clinician.",
        "Review the three component reports separately, especially quality warnings and EEG confounds.",
        "Repeat any low-quality or temporally mismatched assessment before relying on the combined pattern.",
      ]
    : level === "Moderate"
      ? [
          "Discuss the mixed multimodal pattern with a qualified healthcare professional if symptoms or clinical concerns are present.",
          "Review disagreements between modalities instead of relying only on the fused score.",
          "Repeat limited-quality inputs under standardized conditions when practical.",
        ]
      : [
          "Continue routine health monitoring; a Low research score cannot exclude neurological disease.",
          "Seek clinical advice for new or worsening speech, memory, movement, sensory, or sleep symptoms.",
          "Keep component assessments from the same participant and time period when repeating fusion.",
        ];

  return {
    generatedAt: new Date().toISOString(),
    conditions,
    highestCondition,
    highestScore,
    riskLevel: level,
    agreement,
    agreeingModalities,
    headline,
    explanation,
    suggestions,
    cautions,
    sourceIds: {
      Voice: selected.Voice.id,
      General: selected.General.id,
      EEG: selected.EEG.id,
    },
  };
}

export function fusionCandidates(items: HistoryItem[], type: AssessmentType): HistoryItem[] {
  return items.filter((item) => item.type === type && (type === "EEG" ? Boolean(item.eegReport) : Boolean(item.prediction)));
}
