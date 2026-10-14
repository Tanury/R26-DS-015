"use client";

import { CircleAlert, CircleCheck, CircleMinus, Info } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { compareBandProfile, rowScale } from "@/lib/band-pattern";
import type { BandReference } from "@/lib/eeg-types";

/**
 * The subject's band power against the cohort medians for controls and for the
 * highest-scoring condition.
 *
 * Two rules this panel exists to obey. It is labelled descriptive, because the
 * encoder consumes raw epochs and never sees these values — "your recording looks
 * like the AD group's" is true, "this drove your score" is not. And when the backend
 * reports `has_signature: false` it shows the absence instead of the comparison,
 * because on this cohort MS has no band-power pattern and inventing one would turn a
 * 33-year age gap into a neurological finding.
 */

const TONE = {
  match: { icon: CircleCheck, box: "border-blue-200 bg-blue-50", text: "text-blue-900" },
  partial: { icon: Info, box: "border-amber-200 bg-amber-50", text: "text-amber-900" },
  unlike: { icon: CircleMinus, box: "border-slate-200 bg-slate-50", text: "text-slate-700" },
  none: { icon: CircleAlert, box: "border-slate-300 bg-slate-100", text: "text-slate-700" },
} as const;

const LABEL: Record<string, string> = {
  delta: "Delta",
  theta: "Theta",
  alpha: "Alpha",
  beta: "Beta",
  low_gamma: "Low gamma",
  theta_alpha_ratio: "Theta / alpha",
};

export function BandPatternComparison({
  profile,
  reference,
  condition,
}: {
  profile: Record<string, number>;
  reference: BandReference | null;
  condition: string;
}) {
  if (!reference) return null;
  const comparison = compareBandProfile(profile, reference, condition);
  if (!comparison) return null;

  const tone = TONE[comparison.tone];
  const ToneIcon = tone.icon;
  const position = (value: number, scale: number) =>
    `${Math.min((value / scale) * 100, 100)}%`;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3">
        <h2 className="section-title">Band Pattern vs Cohort</h2>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
          descriptive
        </span>
      </CardHeader>
      <CardContent>
        <div className={`flex gap-3 rounded-lg border p-4 ${tone.box}`}>
          <ToneIcon className={`mt-0.5 size-5 shrink-0 ${tone.text}`} />
          <div>
            <div className={`text-sm font-bold ${tone.text}`}>{comparison.headline}</div>
            <p className="mt-1.5 text-xs leading-5 text-slate-600">{comparison.note}</p>
          </div>
        </div>

        {comparison.hasSignature && (
          <>
            <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full bg-blue-700" /> this subject
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full bg-slate-400" /> healthy median
                (n={reference.healthy.n})
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full border-2 border-violet-600" />
                {condition} median (n={reference.conditions[condition]?.n ?? 0})
              </span>
            </div>

            <div className="mt-4 space-y-3.5">
              {comparison.deviations.map((deviation) => {
                const scale = rowScale(deviation);
                return (
                <div key={deviation.band}>
                  <div className="mb-1.5 flex items-baseline gap-2 text-xs">
                    <span className="w-24 shrink-0 font-semibold text-slate-700">
                      {LABEL[deviation.band] ?? deviation.band}
                    </span>
                    {deviation.separates ? (
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                          deviation.matchesCondition
                            ? "bg-blue-100 text-blue-800"
                            : "bg-slate-100 text-slate-600"
                        }`}
                      >
                        {condition} runs {deviation.direction} · AUC{" "}
                        {deviation.auc.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-[10px] text-slate-400">
                        does not separate (AUC {deviation.auc.toFixed(2)})
                      </span>
                    )}
                    <span className="ml-auto tabular-nums font-medium text-slate-700">
                      {deviation.subject.toFixed(3)}
                    </span>
                  </div>

                  <div className="relative h-6 rounded-md bg-slate-100">
                    {/* Group markers first so the subject dot always sits on top. */}
                    <span
                      className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400"
                      style={{ left: position(deviation.healthyMedian, scale) }}
                      title={`Healthy median ${deviation.healthyMedian.toFixed(3)}`}
                    />
                    <span
                      className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-violet-600 bg-white"
                      style={{ left: position(deviation.conditionMedian, scale) }}
                      title={`${condition} median ${deviation.conditionMedian.toFixed(3)}`}
                    />
                    <span
                      className={`absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-white ${
                        deviation.separates && deviation.matchesCondition
                          ? "bg-blue-700"
                          : "bg-slate-600"
                      }`}
                      style={{ left: position(deviation.subject, scale) }}
                      title={`This subject ${deviation.subject.toFixed(3)}`}
                    />
                  </div>
                </div>
                );
              })}
            </div>
          </>
        )}

        <p className="mt-5 border-t border-slate-200 pt-4 text-xs leading-5 text-slate-500">
          {reference.method} A band separates when its AUC against controls clears{" "}
          {reference.separation_margin.toFixed(2)} either side of chance.
        </p>
      </CardContent>
    </Card>
  );
}
