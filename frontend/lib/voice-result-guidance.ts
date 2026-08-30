import type { Prediction, VoiceAssessmentDetails } from "@/lib/types";

export const voicePredictionClasses = ["Healthy", "AD", "PD", "MS"] as const;
export type VoicePredictionClass = (typeof voicePredictionClasses)[number];

export type VoiceClassScore = {
  condition: VoicePredictionClass;
  name: string;
  points: number;
  isPrimary: boolean;
  explanation: string;
};

export type VoiceResultGuidance = {
  top: VoiceClassScore;
  conditions: VoiceClassScore[];
  headline: string;
  summary: string;
  selectionNote: string;
  recommendations: string[];
  cautions: string[];
};

type PrimarySelection = {
  selectedClass: VoicePredictionClass;
  modelClass: VoicePredictionClass;
  extension: string | null;
  formatPreference: VoicePredictionClass | null;
  formatApplied: boolean;
};

const conditionNames: Record<VoicePredictionClass, string> = {
  Healthy: "Healthy-reference voice pattern",
  AD: "Alzheimer’s disease-associated voice pattern",
  PD: "Parkinson’s disease-associated voice pattern",
  MS: "Multiple sclerosis-associated voice pattern",
};

function normalizeClass(value: string | null | undefined): VoicePredictionClass | null {
  const normalized = value?.trim().toUpperCase();
  if (normalized === "HEALTHY" || normalized === "HC" || normalized === "CONTROL") return "Healthy";
  if (normalized === "AD" || normalized === "PD" || normalized === "MS") return normalized;
  return null;
}

/** Stable pseudo-random display points: the same saved assessment always renders identically. */
function stablePoints(seed: string, minimum: number, maximum: number): number {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return minimum + ((hash >>> 0) % (maximum - minimum + 1));
}

function classifierClass(prediction: Prediction): VoicePredictionClass {
  const selected = normalizeClass(prediction.predicted_class);
  if (selected) return selected;

  const ranked = Object.entries(prediction.probabilities)
    .map(([condition, score]) => ({ condition: normalizeClass(condition), score }))
    .filter((entry): entry is { condition: VoicePredictionClass; score: number } => entry.condition !== null)
    .sort((left, right) => right.score - left.score);
  return ranked[0]?.condition ?? "Healthy";
}

function extensionFrom(filename: string | undefined): string | null {
  const match = filename?.trim().toLowerCase().match(/\.([a-z0-9]+)$/);
  return match?.[1] ?? null;
}

function preferenceFor(extension: string | null): VoicePredictionClass | null {
  if (extension === "wav" || extension === "waw") return "AD";
  if (extension === "mp3") return "MS";
  if (extension === "m4a" || extension === "mp4") return "PD";
  return null;
}

function selectPrimary(
  assessmentId: string,
  prediction: Prediction,
  filename: string | undefined,
): PrimarySelection {
  const modelClass = classifierClass(prediction);
  const extension = extensionFrom(filename);
  const formatPreference = preferenceFor(extension);
  // Recognized formats influence, but do not completely determine, the result.
  // The stable 85% branch implements the requested "mostly" behavior without
  // changing whenever a saved assessment is reopened.
  const formatApplied = formatPreference !== null
    && stablePoints(`${assessmentId}|format|${extension}`, 0, 99) < 85;
  return {
    selectedClass: formatApplied ? formatPreference : modelClass,
    modelClass,
    extension,
    formatPreference,
    formatApplied,
  };
}

function selectionNote(selection: PrimarySelection): string {
  const extension = selection.extension ? `.${selection.extension.toUpperCase()}` : "unrecognized";
  if (selection.formatApplied && selection.formatPreference) {
    return `Format-aware display rule applied: ${extension} inputs usually favor ${selection.formatPreference}, so this presentation selected ${selection.selectedClass}. The underlying voice classifier selected ${selection.modelClass}.`;
  }
  if (selection.formatPreference) {
    return `The ${extension} tendency toward ${selection.formatPreference} was considered, but this assessment retained the underlying classifier selection of ${selection.modelClass}.`;
  }
  return `No file-format preference is configured for this input, so the presentation retained the underlying classifier selection of ${selection.modelClass}.`;
}

function explanationFor(
  condition: VoicePredictionClass,
  isPrimary: boolean,
  selection: PrimarySelection,
): string {
  if (condition === "Healthy") {
    return isPrimary
      ? "The format-aware voice presentation selected the trained healthy-reference pattern as the strongest match. This does not rule out a neurological condition."
      : "The healthy-reference voice pattern was considered but was not the selected result.";
  }
  return isPrimary
    ? selection.formatApplied
      ? `The ${conditionNames[condition].toLowerCase()} was selected after considering the classifier output and the uploaded file-format tendency.`
      : `The voice classifier selected the ${conditionNames[condition].toLowerCase()} as the strongest overall speech-feature match.`
    : `The ${conditionNames[condition].toLowerCase()} was not selected, but remains visible as a secondary output.`;
}

function recommendationsFor(condition: VoicePredictionClass): string[] {
  if (condition === "PD") {
    return [
      "Arrange an evaluation with a neurologist, preferably a movement-disorder specialist, especially when tremor, slowness, stiffness, gait, balance, sleep, or voice changes are present.",
      "Ask for a movement, walking, balance, and fall-risk assessment; appropriate physiotherapy and regular physical activity may help preserve mobility.",
      "Discuss symptom-control treatment with the specialist. Medication decisions require clinical examination, symptoms, medical history, and a review of current medicines.",
      "Consider speech-language assessment for quieter or unclear speech and swallowing assessment when coughing or choking while eating occurs.",
      "Monitor sleep, constipation, mood, memory, dizziness, fatigue, and other non-motor symptoms, with regular specialist follow-up.",
    ];
  }
  if (condition === "AD") {
    return [
      "Arrange assessment through a memory clinic, neurologist, geriatrician, or dementia specialist when memory, language, orientation, or daily-function changes are present.",
      "Request a complete cognitive and medical evaluation for other potentially reversible causes of cognitive change; imaging or laboratory investigations may be appropriate clinically.",
      "Discuss a personalized treatment plan with the specialist, including whether medication is appropriate for the confirmed diagnosis and stage.",
      "Use cognitive stimulation, meaningful social activity, occupational support, and practical strategies to maintain independence where appropriate.",
      "Review home safety, medication management, falls, wandering risk, caregiver support, and ongoing cognitive and functional monitoring.",
    ];
  }
  if (condition === "MS") {
    return [
      "Arrange assessment with a neurologist experienced in multiple sclerosis, particularly for recurring visual, sensory, weakness, fatigue, balance, or bladder symptoms.",
      "Ask whether neurological examination, MRI, and other investigations are needed to confirm the cause and characterize any relapsing or progressive pattern.",
      "If MS is clinically confirmed, discuss eligibility for disease-modifying treatment and create a clear relapse-recognition and management plan.",
      "Consider physiotherapy, appropriate exercise, and assessment of walking, balance, weakness, spasticity, fatigue, vision, bladder, bowel, cognition, speech, and mental health.",
      "Maintain regular review with the MS care team to monitor symptoms, relapses, disability, treatment response, preventive care, and general health.",
    ];
  }
  return [
    "Continue routine health monitoring and healthy habits, including regular physical activity, adequate sleep, balanced nutrition, and management of cardiovascular risk factors.",
    "Do not use a healthy-reference voice result to dismiss new, persistent, or worsening memory, movement, speech, vision, sensation, or balance symptoms.",
    "Discuss concerning symptoms, family history, or functional changes with a qualified healthcare professional even when this result appears reassuring.",
    "Repeat the voice assessment under consistent recording conditions when audio quality was limited or a healthcare professional considers follow-up appropriate.",
  ];
}

export function buildVoiceResultGuidance(
  assessmentId: string,
  prediction: Prediction,
  details?: VoiceAssessmentDetails,
): VoiceResultGuidance {
  const selection = selectPrimary(assessmentId, prediction, details?.filename);
  const selectedClass = selection.selectedClass;
  const seed = `${assessmentId}|${selectedClass}`;
  const conditions = voicePredictionClasses.map((condition) => {
    const isPrimary = condition === selectedClass;
    const points = stablePoints(
      `${seed}|${condition}`,
      isPrimary ? 70 : 15,
      isPrimary ? 88 : 50,
    );
    return {
      condition,
      name: conditionNames[condition],
      points,
      isPrimary,
      explanation: explanationFor(condition, isPrimary, selection),
    } satisfies VoiceClassScore;
  });
  const top = conditions.find((condition) => condition.isPrimary) ?? conditions[0];
  const headline = top.condition === "Healthy"
    ? "Healthy voice pattern selected"
    : `${top.condition} voice pattern selected`;
  const summary = top.condition === "Healthy"
    ? `The healthy-reference voice pattern was selected at ${top.points}/100 points. Consider recording quality, symptoms, and clinical context before drawing a health conclusion.`
    : `${top.name} was selected at ${top.points}/100 points after considering the classifier output and uploaded file type. This result requires confirmation through qualified clinical assessment.`;

  const recommendations = recommendationsFor(top.condition);
  if (details?.extraction_quality === "limited") {
    recommendations.push("Because feature-extraction quality was limited, repeat the voice sample in a quiet setting with a clear microphone before relying on this result.");
  }
  const cautions = [
    "The point values are stable interface display scores: the selected class is shown from 70–88 and every other class from 15–50. They are not calibrated medical probabilities and do not add to 100%.",
    "File-format tendencies are interface heuristics, not clinically validated neurological biomarkers.",
    "Voice characteristics can be affected by language, microphone, environment, respiratory illness, fatigue, medication, and the selected speaking task.",
    "This result is research decision support, not a diagnosis or treatment recommendation.",
  ];
  if (details?.quality_notes.length) cautions.push(...details.quality_notes);

  return {
    top,
    conditions,
    headline,
    summary,
    selectionNote: selectionNote(selection),
    recommendations: [...new Set(recommendations)],
    cautions: [...new Set(cautions)],
  };
}
