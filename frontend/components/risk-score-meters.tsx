import Link from "next/link";
import type { ConditionRisk, RiskBand } from "@/lib/eeg-types";
import { isFlaggedSeverity } from "@/lib/eeg-types";

/**
 * Independent risk meters.
 *
 * This deliberately does NOT reuse ProbabilityBars. That component renders a
 * softmax distribution; these are three independent sigmoids that do not sum to 1.
 * A subject can legitimately show elevated risk for more than one condition, and a
 * distribution chart cannot express that without implying the scores compete.
 */

const BAND_STYLE: Record<RiskBand, { bar: string; chip: string }> = {
  Low: { bar: "bg-emerald-600", chip: "bg-emerald-100 text-emerald-800" },
  Medium: { bar: "bg-amber-500", chip: "bg-amber-100 text-amber-800" },
  High: { bar: "bg-red-600", chip: "bg-red-100 text-red-800" },
};

const ORDER = ["AD", "PD", "MS"];

export function RiskScoreMeters({
  conditions,
  highest,
}: {
  conditions: Record<string, ConditionRisk>;
  highest?: string;
}) {
  const names = [
    ...ORDER.filter((name) => name in conditions),
    ...Object.keys(conditions).filter((name) => !ORDER.includes(name)),
  ];
  const total = names.reduce((sum, name) => sum + conditions[name].risk_score, 0);

  return (
    <div className="space-y-6">
      <p className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-600">
        These are <strong>independent</strong> risk scores, not a probability distribution.
        They sum to {Math.round(total * 100)}% rather than 100% — a recording can show an
        elevated pattern for more than one condition at the same time.
      </p>

      {names.map((name) => {
        const condition = conditions[name];
        const style = BAND_STYLE[condition.risk_band] ?? BAND_STYLE.Low;
        const flagged = isFlaggedSeverity(condition.confound_severity);
        const spread = Math.round(condition.epoch_score_std * 100);
        const isTop = highest === name;

        return (
          <div
            key={name}
            className={
              isTop
                ? "rounded-lg border border-blue-200 bg-blue-50/40 p-4"
                : "rounded-lg border border-transparent p-4"
            }
          >
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-semibold text-slate-900">{condition.label}</span>
              <span className="flex items-center gap-2">
                {flagged && (
                  <Link
                    href="/model-card"
                    className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 hover:bg-amber-200"
                    title="This condition carries a documented confound. See the model card."
                  >
                    {condition.confound_severity}
                  </Link>
                )}
                <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${style.chip}`}>
                  {condition.risk_band}
                </span>
                <span className="w-10 text-right text-lg font-bold tabular-nums text-slate-900">
                  {Math.round(condition.risk_score * 100)}
                </span>
              </span>
            </div>

            <div
              className="h-3 overflow-hidden rounded-full bg-slate-200"
              role="meter"
              aria-valuenow={Math.round(condition.risk_score * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${condition.label} risk score`}
            >
              <div
                className={`h-full rounded-full ${style.bar}`}
                style={{ width: `${Math.max(condition.risk_score * 100, 1.5)}%` }}
              />
            </div>

            <div className="mt-1.5 flex flex-wrap justify-between gap-x-4 text-xs text-slate-500">
              <span>epoch spread ±{spread} pts</span>
              {condition.epoch_score_range.length === 2 && (
                <span>
                  range {Math.round(condition.epoch_score_range[0] * 100)}–
                  {Math.round(condition.epoch_score_range[1] * 100)}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
