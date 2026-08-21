import type { BandReference, ConditionBandProfile } from "@/lib/eeg-types";

/**
 * Place one subject's band power against the cohort it came from.
 *
 * This is descriptive. The encoder reads raw time-domain epochs and never sees a
 * spectrum, so nothing here explains a score — it says "your recording looks like
 * the AD group's recordings", which is a different and weaker claim. The occlusion
 * map is the only causal attribution available.
 *
 * The backend owns whether a condition has a band-power signature at all
 * (`ConditionBandProfile.has_signature`); this module only positions the subject
 * within it, and stays silent when the backend says there is nothing to point at.
 */

export type BandDeviation = {
  band: string;
  subject: number;
  healthyMedian: number;
  conditionMedian: number;
  auc: number;
  /** "higher" / "lower" relative to controls, or "none" below the margin. */
  direction: string;
  separates: boolean;
  /** Whether the subject sits on the condition's side of the healthy median. */
  matchesCondition: boolean;
};

export type BandComparison = {
  condition: string;
  hasSignature: boolean;
  deviations: BandDeviation[];
  /** Separating bands on which this subject matches the condition. */
  matched: string[];
  headline: string;
  note: string;
  tone: "match" | "partial" | "unlike" | "none";
};

/**
 * Full-width scale for one band's track.
 *
 * Per row, not shared: the five bands are proportions bounded near 0.5 while
 * theta/alpha is an unbounded ratio that reaches 1.3 here. One shared scale squashes
 * every proportion into the left quarter of its track. Nothing is lost by separating
 * them — each row shows where the subject sits between two group medians, and those
 * positions are only ever compared within a row.
 */
export function rowScale(deviation: BandDeviation): number {
  return (
    Math.max(deviation.subject, deviation.healthyMedian, deviation.conditionMedian, 0.01) *
    1.2
  );
}

export function compareBandProfile(
  profile: Record<string, number>,
  reference: BandReference,
  condition: string,
): BandComparison | null {
  const target: ConditionBandProfile | undefined = reference.conditions[condition];
  if (!target) return null;

  const deviations: BandDeviation[] = reference.bands
    .filter((band) => typeof profile[band] === "number")
    .map((band) => {
      const subject = profile[band];
      const healthyMedian = reference.healthy.medians[band] ?? 0;
      const conditionMedian = target.medians[band] ?? 0;
      const direction = target.direction_vs_hc[band] ?? "none";
      // "On the condition's side" is decided by the group's own direction, not by
      // which median happens to be nearer — a subject can overshoot the condition
      // median and still be moving the right way.
      const matchesCondition =
        direction === "higher"
          ? subject > healthyMedian
          : direction === "lower"
            ? subject < healthyMedian
            : false;
      return {
        band,
        subject,
        healthyMedian,
        conditionMedian,
        auc: target.auc_vs_hc[band] ?? 0.5,
        direction,
        separates: direction !== "none",
        matchesCondition,
      };
    });

  const separating = deviations.filter((d) => d.separates);
  const matched = separating.filter((d) => d.matchesCondition).map((d) => d.band);

  if (!target.has_signature) {
    return {
      condition,
      hasSignature: false,
      deviations,
      matched: [],
      headline: `No band-power pattern to point at for ${condition}`,
      note: target.note,
      tone: "none",
    };
  }

  const ratio = separating.length ? matched.length / separating.length : 0;
  const tone: BandComparison["tone"] =
    ratio >= 0.67 ? "match" : ratio >= 0.34 ? "partial" : "unlike";
  const headline =
    tone === "match"
      ? `Matches the ${condition} band profile on ${matched.length} of ${separating.length} discriminating bands`
      : tone === "partial"
        ? `Partly matches the ${condition} band profile — ${matched.length} of ${separating.length} discriminating bands`
        : `Does not match the ${condition} band profile (${matched.length} of ${separating.length} discriminating bands)`;

  return {
    condition,
    hasSignature: true,
    deviations,
    matched,
    headline,
    note: target.note,
    tone,
  };
}
