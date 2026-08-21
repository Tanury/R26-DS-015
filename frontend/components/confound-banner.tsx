import Link from "next/link";
import { AlertTriangle, Info } from "lucide-react";
import type { ConfoundDisclosure } from "@/lib/eeg-types";

/**
 * Render-blocking confound disclosure.
 *
 * The training cohort has a documented age and site confound: MS participants were
 * ~33 years younger than every other group and all came from one recording site.
 * When a displayed condition carries CRITICAL severity this banner is persistent and
 * NOT dismissible — shipping the number without the caveat would be misleading by
 * omission. All copy comes from the model card so it stays true after a retrain.
 */
export function ConfoundBanner({ disclosure }: { disclosure: ConfoundDisclosure }) {
  const critical = Object.entries(disclosure.severity_by_condition).filter(([, severity]) =>
    severity.toUpperCase().includes("CRITICAL"),
  );
  const moderate = Object.entries(disclosure.severity_by_condition).filter(([, severity]) => {
    const value = severity.toUpperCase();
    return !value.includes("CRITICAL") && (value.includes("HIGH") || value.includes("MODERATE"));
  });

  if (!critical.length && !moderate.length) {
    return (
      <div className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
        <Info className="mt-0.5 size-5 shrink-0 text-slate-500" />
        <p>No class-level confound was flagged for this model.</p>
      </div>
    );
  }

  const mae = disclosure.age_probe_mae_years;
  const improvement = disclosure.age_probe_improvement_over_baseline;

  return (
    <section
      role="note"
      aria-label="Confound disclosure"
      className="rounded-lg border-2 border-amber-300 bg-amber-50 p-5"
    >
      <div className="flex gap-3">
        <AlertTriangle className="mt-0.5 size-6 shrink-0 text-amber-700" />
        <div className="min-w-0 space-y-3">
          <h2 className="text-base font-bold text-amber-900">
            {critical.length > 0
              ? `Interpret the ${critical.map(([name]) => name).join(" and ")} score${critical.length > 1 ? "s" : ""} with caution`
              : "This model carries documented confounds"}
          </h2>

          <p className="text-sm leading-6 text-amber-900">{disclosure.statement}</p>

          <div className="flex flex-wrap gap-2">
            {critical.map(([name, severity]) => (
              <span
                key={name}
                className="rounded-full bg-red-100 px-3 py-1 text-xs font-bold text-red-800"
              >
                {name}: {severity}
              </span>
            ))}
            {moderate.map(([name, severity]) => (
              <span
                key={name}
                className="rounded-full bg-amber-200 px-3 py-1 text-xs font-semibold text-amber-900"
              >
                {name}: {severity}
              </span>
            ))}
          </div>

          <dl className="grid gap-3 text-xs text-amber-900 sm:grid-cols-3">
            {mae !== null && (
              <div>
                <dt className="font-semibold">Age recoverable from z_eeg</dt>
                <dd className="mt-0.5">
                  within {mae.toFixed(1)} years
                  {improvement !== null && ` (${Math.round(improvement * 100)}% better than guessing the mean)`}
                </dd>
              </div>
            )}
            {disclosure.site_probe_balanced_accuracy !== null && (
              <div>
                <dt className="font-semibold">Recording site recoverable</dt>
                <dd className="mt-0.5">
                  {Math.round(disclosure.site_probe_balanced_accuracy * 100)}% balanced accuracy
                </dd>
              </div>
            )}
            {Object.keys(disclosure.risk_score_age_correlation).length > 0 && (
              <div>
                <dt className="font-semibold">Score-vs-age, within negatives</dt>
                <dd className="mt-0.5 tabular-nums">
                  {Object.entries(disclosure.risk_score_age_correlation)
                    .map(([name, r]) => `${name} r=${r.toFixed(2)}`)
                    .join(" · ")}
                </dd>
              </div>
            )}
          </dl>

          <Link
            href="/model-card"
            className="inline-block text-sm font-semibold text-amber-900 underline underline-offset-2 hover:text-amber-700"
          >
            Read the full model card
          </Link>
        </div>
      </div>
    </section>
  );
}
