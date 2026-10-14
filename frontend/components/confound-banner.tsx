import Link from "next/link";
import { AlertTriangle, ArrowRight, BarChart3, Info, MapPin } from "lucide-react";
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
export function ConfoundBanner({
  disclosure,
  showModelCardLink = true,
}: {
  disclosure: ConfoundDisclosure;
  showModelCardLink?: boolean;
}) {
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
  const flaggedConditions = [...critical, ...moderate].map(([name]) => name);
  const formattedCorrelations = Object.entries(disclosure.risk_score_age_correlation)
    .map(([name, r]) => `${name} ${r >= 0 ? "+" : ""}${r.toFixed(2)}`)
    .join(" · ");

  return (
    <section
      role="note"
      aria-label="Confound disclosure"
      className="overflow-hidden rounded-xl border border-amber-300 bg-amber-50"
    >
      <div className="border-b border-amber-200 bg-amber-100/70 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <div className="grid size-10 shrink-0 place-items-center rounded-full bg-amber-200">
            <AlertTriangle className="size-5 text-amber-800" />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-bold uppercase tracking-wide text-amber-800">
              Important model limitation
            </p>
            <h2 className="mt-1 text-lg font-bold text-slate-950">
              {flaggedConditions.join(" and ")} score{flaggedConditions.length > 1 ? "s" : ""} may reflect training-dataset differences
            </h2>
            <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-700">
              Age and recording-site patterns may influence this model&apos;s output. Treat the affected score as a research signal—not proof of a neurological condition—and do not use it alone for a medical decision.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-5 p-5 sm:p-6">
        <div>
          <div className="text-xs font-bold uppercase tracking-wide text-slate-500">
            Condition-level caution
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(disclosure.severity_by_condition).map(([name, severity]) => {
              const flagged = severity.toUpperCase().includes("CRITICAL")
                || severity.toUpperCase().includes("HIGH")
                || severity.toUpperCase().includes("MODERATE");
              return (
                <span
                  key={name}
                  className={flagged
                    ? "rounded-full bg-amber-200 px-3 py-1.5 text-xs font-bold text-amber-950"
                    : "rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600"}
                >
                  {name}: {severity}
                </span>
              );
            })}
          </div>
        </div>

        <dl className="grid gap-3 lg:grid-cols-3">
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="flex items-center gap-2">
              <Info className="size-4 text-blue-700" />
              <dt className="text-sm font-bold text-slate-900">Age check</dt>
            </div>
            <dd className="mt-2 text-sm font-semibold text-slate-800">
              {mae === null ? "Unavailable during training" : `Recoverable within ${mae.toFixed(1)} years`}
            </dd>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {mae === null
                ? "Age metadata was missing, so the embedding age-bias probe could not run."
                : `${improvement !== null ? `${Math.round(improvement * 100)}% better than the baseline. ` : ""}Age information remains detectable in the representation.`}
            </p>
          </div>

          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="flex items-center gap-2">
              <MapPin className="size-4 text-blue-700" />
              <dt className="text-sm font-bold text-slate-900">Recording-site check</dt>
            </div>
            <dd className="mt-2 text-sm font-semibold text-slate-800">
              {disclosure.site_probe_balanced_accuracy === null
                ? "Not available"
                : `${Math.round(disclosure.site_probe_balanced_accuracy * 100)}% balanced accuracy`}
            </dd>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {disclosure.site_probe_balanced_accuracy === null
                ? "This model card does not include a recording-site probe result."
                : "Recording-site information is detectable, so site or equipment differences may influence the model representation."}
            </p>
          </div>

          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="size-4 text-blue-700" />
              <dt className="text-sm font-bold text-slate-900">Score–age association</dt>
            </div>
            <dd className="mt-2 text-sm font-semibold tabular-nums text-slate-800">
              {formattedCorrelations || "Not available"}
            </dd>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              Correlations are measured within participants negative for each condition. Values nearer zero indicate a weaker linear association.
            </p>
          </div>
        </dl>

        <details className="rounded-lg border border-amber-200 bg-amber-100/40 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-slate-800">
            Technical details and data limitations
          </summary>
          <p className="mt-3 text-sm leading-6 text-slate-700">{disclosure.statement}</p>
        </details>

        {showModelCardLink && (
          <Link
            href="/model-card"
            className="inline-flex items-center gap-2 text-sm font-semibold text-blue-700 hover:text-blue-900 hover:underline"
          >
            Read the full model card
            <ArrowRight className="size-4" />
          </Link>
        )}
      </div>
    </section>
  );
}
