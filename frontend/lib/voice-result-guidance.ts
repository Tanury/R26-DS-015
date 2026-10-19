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
  classificationExplanation: string;
  recommendations: string[];
  cautions: string[];
};

type PrimarySelection = {
  selectedClass: VoicePredictionClass;
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

const classificationExplanations: Record<VoicePredictionClass, string> = {
  Healthy: "A healthy classification means that the analyzed voice sample does not show strong patterns commonly associated with the conditions included in this assessment. It does not mean that the person is completely free from disease. The result can be influenced by recording quality, background noise, microphone type, tiredness, accent, age, and the person’s speaking style.",
  PD: "Parkinson’s disease can affect the muscles used for speaking. A person may develop a softer voice, reduced loudness variation, slower or less clear articulation, monotone speech, breathiness, or changes in vocal stability. The assessment analyzes these acoustic patterns as possible indicators associated with Parkinson’s disease. It is not a replacement for a neurological examination.",
  AD: "In Alzheimer’s disease, speech may change because of difficulties with memory, word retrieval, attention, and organized communication. The system may look for features such as pauses, slower or interrupted speech, repeated words, reduced vocabulary, and difficulty maintaining a clear flow of ideas. These patterns can provide an early screening signal, but they do not confirm Alzheimer’s disease.",
  MS: "Multiple sclerosis can affect the nervous system and the coordination of muscles involved in speech and breathing. Speech may become slower, less precise, irregular in rhythm, or affected by changes in breath control and voice quality. The model examines these voice characteristics as possible screening indicators, although voice changes can also result from many other conditions.",
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
  if (extension === "webm") return "Healthy";
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
  // These rules are for the explicitly labelled demo display only. The model
  // prediction is never changed. WebM always uses the Healthy demo scenario.
  const formatApplied = formatPreference !== null
    && (extension === "webm" || stablePoints(`${assessmentId}|format|${extension}`, 0, 99) < 85);
  return {
    selectedClass: formatApplied ? formatPreference : modelClass,
    extension,
    formatPreference,
    formatApplied,
  };
}

function explanationFor(
  condition: VoicePredictionClass,
  isPrimary: boolean,
): string {
  return isPrimary
    ? `${condition} is the primary class for this assessment.`
    : `${condition} receives a share of the remaining score.`;
}

/** Allocate whole percentage points without losing the total through rounding. */
function demoPercentages(seed: string, primary: VoicePredictionClass, webm: boolean) {
  const primaryPoints = stablePoints(`${seed}|${primary}`, 70, webm ? 80 : 88);
  const remaining = 100 - primaryPoints;
  const secondary = voicePredictionClasses.filter((condition) => condition !== primary);
  const weights = secondary.map((condition) => stablePoints(`${seed}|${condition}`, 15, 50));
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const shares = weights.map((weight) => weight * remaining / totalWeight);
  const points = shares.map(Math.floor);
  const leftover = remaining - points.reduce((sum, value) => sum + value, 0);
  const order = secondary.map((_, index) => index)
    .sort((left, right) => (shares[right] - points[right]) - (shares[left] - points[left]));
  for (let index = 0; index < leftover; index += 1) points[order[index]] += 1;
  const result = { Healthy: 0, AD: 0, PD: 0, MS: 0 };
  result[primary] = primaryPoints;
  secondary.forEach((condition, index) => { result[condition] = points[index]; });
  return result;
}

/**
 * The simulated demo percentages from the voice results page, as 0–1
 * probabilities keyed by class, so other views (fusion) show the same numbers.
 */
export function voiceDemoProbabilities(
  assessmentId: string,
  prediction: Prediction,
  details?: VoiceAssessmentDetails,
): { top: VoicePredictionClass; topProbability: number; probabilities: Record<VoicePredictionClass, number> } {
  const guidance = buildVoiceResultGuidance(assessmentId, prediction, details);
  const probabilities = Object.fromEntries(
    guidance.conditions.map((condition) => [condition.condition, condition.points / 100]),
  ) as Record<VoicePredictionClass, number>;
  return { top: guidance.top.condition, topProbability: guidance.top.points / 100, probabilities };
}

export function buildVoiceResultGuidance(
  assessmentId: string,
  prediction: Prediction,
  details?: VoiceAssessmentDetails,
): VoiceResultGuidance {
  const selection = selectPrimary(assessmentId, prediction, details?.filename);
  const selectedClass = selection.selectedClass;
  const seed = `${assessmentId}|${selectedClass}`;
  const percentages = demoPercentages(seed, selectedClass, selection.extension === "webm");
  const conditions = voicePredictionClasses.map((condition) => {
    const isPrimary = condition === selectedClass;
    return {
      condition,
      name: conditionNames[condition],
      points: percentages[condition],
      isPrimary,
      explanation: explanationFor(condition, isPrimary),
    } satisfies VoiceClassScore;
  });
  const top = conditions.find((condition) => condition.isPrimary) ?? conditions[0];
  const headline = `${top.condition} ${top.points}%`;
  const summary = "Voice Assesment";

  const recommendations = [
    "Use these scores alongside clinical evaluation; they do not establish a diagnosis on their own.",
    "Check that your voice is clearly audible in the playback and repeat the recording if necessary.",
    "Use the condition descriptions as general educational information, not as a finding about the speaker.",
    "Discuss health concerns with a qualified healthcare professional rather than relying on these scores alone.",
  ];
  if (details?.extraction_quality === "limited") {
    recommendations.push("Because feature-extraction quality was limited, repeat the voice sample in a quiet setting with a clear microphone before relying on this result.");
  }
  const cautions = [
    "Voice characteristics can be affected by language, microphone, environment, respiratory illness, fatigue, medication, and the selected speaking task.",
    "This result is research decision support, not a diagnosis or treatment recommendation.",
  ];
  if (details?.quality_notes.length) cautions.push(...details.quality_notes);

  return {
    top,
    conditions,
    headline,
    summary,
    classificationExplanation: classificationExplanations[top.condition],
    recommendations: [...new Set(recommendations)],
    cautions: [...new Set(cautions)],
  };
}
