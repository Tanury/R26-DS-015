import type { RiskCondition } from "@/lib/eeg-types";
import type { AssessmentType, HistoryItem, Prediction } from "@/lib/types";
import { voiceDemoProbabilities } from "@/lib/voice-result-guidance";

export const fusionConditions = ["AD", "PD", "MS"] as const satisfies readonly RiskCondition[];
export type FusionCondition = (typeof fusionConditions)[number];
export type FusionModality = "Voice" | "General" | "EEG" | "Neuroimaging";
/** The models that score every condition (each may be left out of fusion). */
export type CoreFusionModality = Exclude<FusionModality, "Neuroimaging">;
export type CoreSelection = Partial<Record<CoreFusionModality, HistoryItem>>;
export type FusionAgreement = "Strong" | "Partial" | "Discordant" | "Single model" | "Not applicable";
/** Optional single-condition imaging scans, at most one per condition. */
export type ImagingSelection = Partial<Record<FusionCondition, HistoryItem>>;
export type FusionRiskLevel = "Low" | "Moderate" | "High";
export type FusionClass = FusionCondition | "HC";
export const FINAL_CLASS_THRESHOLD = 0.5;
export const fusionClassNames: Record<FusionClass, string> = {
  AD: "Alzheimer’s disease (AD)",
  PD: "Parkinson’s disease (PD)",
  MS: "Multiple sclerosis (MS)",
  HC: "Healthy control (HC)",
};

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
  /** False when no included component carries evidence for this condition. */
  informed: boolean;
  contributions: FusionContribution[];
};

export type FusionResult = {
  generatedAt: string;
  conditions: Record<FusionCondition, FusedCondition>;
  highestCondition: FusionCondition;
  highestScore: number;
  /** Single fused label: the top condition, or HC when none reaches the threshold. */
  finalClass: FusionClass;
  riskLevel: FusionRiskLevel;
  agreement: FusionAgreement;
  agreeingModalities: number;
  /** How many included models score every condition (Voice, Biomedical, EEG). */
  rankingModalities: number;
  includedModalities: FusionModality[];
  excludedModalities: FusionModality[];
  headline: string;
  explanation: string;
  suggestions: string[];
  cautions: string[];
  sourceIds: Partial<Record<CoreFusionModality, string>>;
  imagingSourceIds: Partial<Record<FusionCondition, string>>;
  imagingConditions: FusionCondition[];
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

/**
 * Imaging models are trained for one condition each, so their evidence is
 * the disease probability against the normal class. The AD model has a third
 * MCI class; it is counted as half-way between CN and AD rather than dropped.
 */
function imagingEvidence(item: HistoryItem, condition: FusionCondition): number {
  const probabilities = item.neuroimaging?.probabilities ?? {};
  const value = (key: string) => clamp(Number(probabilities[key] ?? 0));
  if (condition === "AD") {
    const total = value("AD") + value("MCI") + value("CN");
    return total > 0 ? clamp((value("AD") + 0.5 * value("MCI")) / total) : 0;
  }
  const disease = value(condition);
  const healthy = value("HC");
  return disease + healthy > 0 ? disease / (disease + healthy) : disease;
}

function imagingWeight(item: HistoryItem): { weight: number; label: string; note: string } {
  const scan = item.neuroimaging!;
  if (scan.demo) {
    return {
      weight: 0.25,
      label: `${scan.scanType} · demo output`,
      note: `The backend had no trained ${scan.condition} checkpoint and returned a demo prediction, so its influence is reduced to 25%.`,
    };
  }
  if (scan.cohort) {
    return {
      weight: 1,
      label: `${scan.scanType} · processed cohort`,
      note: `Precomputed ${scan.condition} vision-encoder result (LOOCV accuracy ${Math.round(scan.cohort.loocvAccuracy * 100)}%, AUC ${scan.cohort.loocvAuc.toFixed(2)}); it contributes only to ${scan.condition}.`,
    };
  }
  return {
    weight: 1,
    label: scan.scanType,
    note: `${scan.condition}-specific imaging model (${scan.modality}); it contributes only to ${scan.condition}.`,
  };
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

function topCondition(scores: Record<FusionCondition, number>, among: readonly FusionCondition[] = fusionConditions): FusionCondition {
  return [...among].sort((left, right) => scores[right] - scores[left])[0];
}

const modalityNames: Record<FusionModality, string> = {
  Voice: "Voice",
  General: "Biomedical",
  EEG: "EEG",
  Neuroimaging: "Neuroimaging",
};

function listNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function latestFusionAssessments(items: HistoryItem[]): Partial<Record<FusionModality, HistoryItem>> {
  const latest: Partial<Record<FusionModality, HistoryItem>> = {};
  for (const item of items) {
    if (!latest[item.type]) latest[item.type] = item;
  }
  return latest;
}

/**
 * Late fusion over whichever components were included. Any component may be
 * left out; at least one Voice, Biomedical, EEG, or Neuroimaging result is
 * required. Excluded components contribute nothing, not a neutral score.
 */
export function buildFusionResult(
  selected: CoreSelection,
  imaging: ImagingSelection = {},
): FusionResult {
  const voice = selected.Voice?.prediction ? selected.Voice : undefined;
  const general = selected.General?.prediction ? selected.General : undefined;
  const eegItem = selected.EEG?.eegReport ? selected.EEG : undefined;
  const eegReport = eegItem?.eegReport;
  const scans = Object.fromEntries(
    fusionConditions.flatMap((condition) => {
      const scan = imaging[condition];
      return scan?.neuroimaging?.condition === condition ? [[condition, scan]] : [];
    }),
  ) as ImagingSelection;
  if (!voice && !general && !eegItem && Object.keys(scans).length === 0) {
    throw new Error("Include at least one Voice, Biomedical, EEG, or Neuroimaging result to calculate a fusion result.");
  }

  const voiceQuality = voice ? voiceWeight(voice) : undefined;
  const biomedicalQuality = general ? biomedicalWeight(general) : undefined;
  const eegBaseWeight = eegReport ? eegQualityWeight(eegReport.signal_quality.grade) : 0;
  const scoresFor = (score: (condition: FusionCondition) => number) =>
    Object.fromEntries(fusionConditions.map((condition) => [condition, score(condition)])) as Record<FusionCondition, number>;
  // The models that score every condition; only these can rank conditions against each other.
  const modalityScores: Partial<Record<CoreFusionModality, Record<FusionCondition, number>>> = {};
  if (voice) {
    // Use the same simulated demo scores the voice results page displays.
    const demo = voiceDemoProbabilities(voice.id, voice.prediction!, voice.voiceDetails);
    const demoPrediction: Prediction = { ...voice.prediction!, probabilities: demo.probabilities };
    modalityScores.Voice = scoresFor((condition) => oneVsHealthyEvidence(demoPrediction, condition));
  }
  if (general) modalityScores.General = scoresFor((condition) => oneVsHealthyEvidence(general.prediction!, condition));
  if (eegReport) {
    modalityScores.EEG = scoresFor((condition) =>
      clamp(eegReport.risk_assessment.conditions[condition]?.risk_score ?? eegReport.risk_scores[condition] ?? 0));
  }
  const eegSeverity = (condition: FusionCondition) => eegReport
    ? eegReport.risk_assessment.conditions[condition]?.confound_severity
      ?? eegReport.confound_disclosure.severity_by_condition[condition]
      ?? "None"
    : "None";

  const conditions = Object.fromEntries(fusionConditions.map((condition) => {
    const contributions: FusionContribution[] = [];
    if (voiceQuality && modalityScores.Voice) {
      contributions.push({
        modality: "Voice",
        evidence: modalityScores.Voice[condition],
        effectiveWeight: voiceQuality.weight,
        qualityLabel: voiceQuality.label,
        note: voiceQuality.note,
      });
    }
    if (biomedicalQuality && modalityScores.General) {
      contributions.push({
        modality: "General",
        evidence: modalityScores.General[condition],
        effectiveWeight: biomedicalQuality.weight,
        qualityLabel: biomedicalQuality.label,
        note: biomedicalQuality.note,
      });
    }
    if (eegReport && modalityScores.EEG) {
      const severity = eegSeverity(condition);
      contributions.push({
        modality: "EEG",
        evidence: modalityScores.EEG[condition],
        effectiveWeight: eegBaseWeight * confoundWeight(severity),
        qualityLabel: `${eegReport.signal_quality.grade} signal · ${severity || "no"} confound`,
        note: `EEG influence combines the ${eegReport.signal_quality.grade} signal-quality factor with the documented ${condition} confound factor.`,
      });
    }
    const scan = scans[condition];
    if (scan) {
      const quality = imagingWeight(scan);
      contributions.push({
        modality: "Neuroimaging",
        evidence: imagingEvidence(scan, condition),
        effectiveWeight: quality.weight,
        qualityLabel: quality.label,
        note: quality.note,
      });
    }
    const denominator = contributions.reduce((sum, contribution) => sum + contribution.effectiveWeight, 0);
    const score = denominator
      ? contributions.reduce((sum, contribution) => sum + contribution.evidence * contribution.effectiveWeight, 0) / denominator
      : 0;
    const value: FusedCondition = {
      condition,
      score: clamp(score),
      level: riskLevel(score),
      informed: contributions.length > 0,
      contributions,
    };
    return [condition, value];
  })) as Record<FusionCondition, FusedCondition>;

  const informedConditions = fusionConditions.filter((condition) => conditions[condition].informed);
  const fusedScores = Object.fromEntries(fusionConditions.map((condition) => [condition, conditions[condition].score])) as Record<FusionCondition, number>;
  const highestCondition = topCondition(fusedScores, informedConditions);
  const highestScore = fusedScores[highestCondition];
  const rankingModalities = Object.keys(modalityScores) as CoreFusionModality[];
  const modalityLeaders = rankingModalities.map((modality) => topCondition(modalityScores[modality]!));
  const agreeingModalities = modalityLeaders.filter((condition) => condition === highestCondition).length;
  const rankingCount = rankingModalities.length;
  const agreement: FusionAgreement = rankingCount === 0
    ? "Not applicable"
    : rankingCount === 1
      ? "Single model"
      : agreeingModalities === rankingCount ? "Strong" : agreeingModalities >= 2 ? "Partial" : "Discordant";
  const ranked = [...informedConditions].sort((left, right) => fusedScores[right] - fusedScores[left]);
  const margin = ranked.length > 1 ? fusedScores[ranked[0]] - fusedScores[ranked[1]] : 0;
  const imagingConditions = fusionConditions.filter((condition) => Boolean(scans[condition]));
  const includedModalities: FusionModality[] = [
    ...rankingModalities,
    ...(imagingConditions.length ? ["Neuroimaging" as const] : []),
  ];
  const excludedModalities = (["Voice", "General", "EEG", "Neuroimaging"] as FusionModality[])
    .filter((modality) => !includedModalities.includes(modality));

  const cautions: string[] = [];
  if (includedModalities.length > 1) {
    cautions.push("The selected assessments are not linked by a shared patient identifier. Verify that they all belong to the same participant and a comparable time period.");
  }
  cautions.push("This is deterministic late fusion of separately trained models, not a jointly trained or clinically calibrated multimodal model.");
  if (excludedModalities.length) {
    cautions.push(`Not included: ${listNames(excludedModalities.map((modality) => modalityNames[modality]))}. The result relies on ${listNames(includedModalities.map((modality) => modalityNames[modality]))} only.`);
  }
  if (includedModalities.length === 1) {
    cautions.push(`Only ${modalityNames[includedModalities[0]]} was included, so this is a single-modality result rather than a multimodal fusion.`);
  }
  const uninformed = fusionConditions.filter((condition) => !conditions[condition].informed);
  if (uninformed.length) {
    cautions.push(`${listNames([...uninformed])} ${uninformed.length === 1 ? "has" : "have"} no evidence from the included components and ${uninformed.length === 1 ? "was" : "were"} not scored.`);
  }
  if (agreement === "Partial" || agreement === "Discordant") {
    cautions.push(`The component leaders are ${modalityLeaders.join(", ")}; the modalities do not fully agree.`);
  }
  const topSeverity = eegSeverity(highestCondition);
  if (eegReport && /CRITICAL|HIGH|MODERATE/i.test(topSeverity)) {
    cautions.push(`The ${highestCondition} EEG contribution carries a documented ${topSeverity} confound and was down-weighted.`);
  }
  const imagingFor = (condition: FusionCondition) =>
    conditions[condition].contributions.find((contribution) => contribution.modality === "Neuroimaging")?.evidence;
  const topImaging = imagingFor(highestCondition);
  if (topImaging !== undefined && topImaging < 0.5) {
    cautions.push(`The ${highestCondition} neuroimaging scan does not support the fused leader (${Math.round(topImaging * 100)}/100 imaging evidence).`);
  }
  for (const condition of imagingConditions) {
    const evidence = imagingFor(condition)!;
    if (condition !== highestCondition && evidence >= 0.5) {
      cautions.push(`The ${condition} neuroimaging scan is elevated (${Math.round(evidence * 100)}/100) although ${condition} is not the highest fused condition.`);
    }
  }
  const demoScans = imagingConditions.filter((condition) => scans[condition]?.neuroimaging?.demo);
  if (demoScans.length) {
    cautions.push(`The ${demoScans.join(", ")} neuroimaging result${demoScans.length === 1 ? " is a demo" : "s are demo"} output from a model without a trained checkpoint and ${demoScans.length === 1 ? "was" : "were"} down-weighted to 25%. Do not interpret ${demoScans.length === 1 ? "it" : "them"} as real imaging evidence.`);
  }
  const inSampleScans = imagingConditions.filter((condition) => scans[condition]?.neuroimaging?.cohort?.inSample);
  if (inSampleScans.length) {
    cautions.push(`The ${inSampleScans.join(", ")} neuroimaging result comes from the processed training cohort; the encoder was fit on that subject, so its prediction is in-sample and optimistic. Use the LOOCV metrics as the realistic performance estimate.`);
  }
  if (imagingConditions.length > 0 && imagingConditions.length < fusionConditions.length && rankingCount > 0) {
    const missing = fusionConditions.filter((condition) => !imagingConditions.includes(condition));
    cautions.push(`Neuroimaging covers ${imagingConditions.join(", ")} only; ${missing.join(", ")} fused scores have no imaging input, so scores are not equally informed across conditions.`);
  }
  if (voice?.voiceDetails?.extraction_quality === "limited") {
    cautions.push("Voice extraction quality was limited, so the voice contribution was down-weighted.");
  }

  const level = riskLevel(highestScore);
  // Fused scores are disease-vs-healthy evidence, so 0.5 is the even-odds point.
  const finalClass: FusionClass = highestScore >= FINAL_CLASS_THRESHOLD ? highestCondition : "HC";
  if (finalClass === "HC" && level !== "Low") {
    cautions.push(`Borderline: ${highestCondition} is ${Math.round(highestScore * 100)}/100, just below the ${FINAL_CLASS_THRESHOLD * 100}/100 classification threshold.`);
  }
  const headline = `Final fused classification: ${fusionClassNames[finalClass]}`;
  const imagingNote = topImaging === undefined
    ? ""
    : ` The ${highestCondition} neuroimaging scan ${topImaging >= 0.5 ? "supports" : "does not support"} this pattern at ${Math.round(topImaging * 100)}/100.`;
  const agreementNote = rankingCount > 1
    ? ` ${agreeingModalities} of ${rankingCount} component models rank ${highestCondition} first.`
    : "";
  const marginNote = ranked.length > 1 ? ` It leads the next fused condition by ${Math.round(margin * 100)} points.` : "";
  const explanation = finalClass === "HC"
    ? `No condition reaches the ${FINAL_CLASS_THRESHOLD * 100}/100 classification threshold. ${highestCondition} is the highest at ${Math.round(highestScore * 100)}/100, but this does not rule out disease.${imagingNote}`
    : `${highestCondition} has the highest quality-adjusted fused score at ${Math.round(highestScore * 100)}/100.${agreementNote}${marginNote}${imagingNote}`;
  const suggestions = level === "High"
    ? [
        "Arrange timely review with a qualified neurologist or appropriate clinician.",
        "Review the component reports separately, especially quality warnings and EEG confounds.",
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
    finalClass,
    riskLevel: level,
    agreement,
    agreeingModalities,
    rankingModalities: rankingCount,
    includedModalities,
    excludedModalities,
    headline,
    explanation,
    suggestions,
    cautions,
    sourceIds: Object.fromEntries(
      rankingModalities.map((modality) => [modality, selected[modality]!.id]),
    ) as Partial<Record<CoreFusionModality, string>>,
    imagingSourceIds: Object.fromEntries(
      imagingConditions.map((condition) => [condition, scans[condition]!.id]),
    ) as Partial<Record<FusionCondition, string>>,
    imagingConditions,
  };
}

export function fusionCandidates(items: HistoryItem[], type: AssessmentType): HistoryItem[] {
  return items.filter((item) => item.type === type && (
    type === "EEG" ? Boolean(item.eegReport)
      : type === "Neuroimaging" ? Boolean(item.neuroimaging)
        : Boolean(item.prediction)
  ));
}

export function imagingCandidates(items: HistoryItem[], condition: FusionCondition): HistoryItem[] {
  return fusionCandidates(items, "Neuroimaging").filter((item) => item.neuroimaging?.condition === condition);
}
