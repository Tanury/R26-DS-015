"use client";

import { useEffect, useState } from "react";
import { Ban, Cpu, ServerCrash, Target } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { ConfoundBanner } from "@/components/confound-banner";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { fetchModelCard } from "@/lib/eeg-api";
import type { EegModelCard } from "@/lib/eeg-types";

type ConditionPerformance = { auc: number | null; auc_ci?: { ci_low: number; ci_high: number } | null };

export default function ModelCardPage() {
  const [card, setCard] = useState<EegModelCard | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetchModelCard()
      .then((result) => {
        if (!cancelled) setCard(result);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Unable to load the model card.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <AppShell>
        <div className="mx-auto max-w-xl py-20 text-center">
          <ServerCrash className="mx-auto size-12 text-red-700" />
          <h1 className="mt-5 text-2xl font-bold">Model card unavailable</h1>
          <p className="mt-2 text-slate-600">{error}</p>
        </div>
      </AppShell>
    );
  }

  if (!card) {
    return (
      <AppShell>
        <p className="py-20 text-center text-sm text-slate-500">Loading model card…</p>
      </AppShell>
    );
  }

  const pooled = (card.performance.pooled_per_condition ?? {}) as Record<string, ConditionPerformance>;
  const cohort = card.cohort as Record<string, unknown>;
  const perClass = (cohort.per_class ?? {}) as Record<string, number>;
  const perSite = (cohort.per_site ?? {}) as Record<string, number>;
  const outOfScope = (card.intended_use.out_of_scope ?? []) as string[];

  return (
    <AppShell>
      <PageHeader
        title="EEG Model Card"
        description="What this encoder is, what it was trained on, how well it performs, and — most importantly — what its numbers cannot be taken to mean."
      />

      <div className="space-y-6">
        <ConfoundBanner disclosure={card.confound_disclosure} />

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader className="flex flex-row items-center gap-3">
              <Cpu className="size-5 text-blue-700" />
              <h2 className="section-title">Architecture</h2>
            </CardHeader>
            <CardContent>
              <dl className="space-y-3 text-sm">
                {[
                  ["Encoder", card.architecture],
                  ["Input representation", card.input_representation],
                  ["Input shape", `[${card.input_shape.join(", ")}]`],
                  ["Embedding", `${card.embedding_dim}-D, L2-normalized`],
                  ["Risk heads", card.risk_conditions.join(" · ") + " (independent sigmoids)"],
                  ["Run id", card.run_id],
                  ["Inference enabled", card.inference_available ? "yes" : "no — cohort browsing only"],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between gap-4 border-b border-slate-100 pb-2">
                    <dt className="font-semibold text-slate-600">{label}</dt>
                    <dd className="text-right">{value}</dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h2 className="section-title">Training cohort</h2>
            </CardHeader>
            <CardContent>
              <dl className="space-y-3 text-sm">
                <div className="flex justify-between gap-4 border-b border-slate-100 pb-2">
                  <dt className="font-semibold text-slate-600">Dataset</dt>
                  <dd className="text-right">{String(cohort.dataset ?? "—")}</dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-slate-100 pb-2">
                  <dt className="font-semibold text-slate-600">Subjects</dt>
                  <dd className="text-right">{String(cohort.n_subjects ?? "—")}</dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-slate-100 pb-2">
                  <dt className="font-semibold text-slate-600">Per class</dt>
                  <dd className="text-right">
                    {Object.entries(perClass).map(([k, v]) => `${k} ${v}`).join(" · ") || "—"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-slate-100 pb-2">
                  <dt className="font-semibold text-slate-600">Per site</dt>
                  <dd className="text-right">
                    {Object.entries(perSite).map(([k, v]) => `${k} ${v}`).join(" · ") || "—"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="font-semibold text-slate-600">Validation</dt>
                  <dd className="text-right">{String(cohort.cross_validation ?? "—")}</dd>
                </div>
              </dl>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader className="flex flex-row items-center gap-3">
            <Target className="size-5 text-blue-700" />
            <h2 className="section-title">Performance</h2>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-sm">
                <thead className="text-xs uppercase text-slate-500">
                  <tr>
                    {["Risk score", "AUC", "95% CI", "Confound"].map((heading) => (
                      <th key={heading} className="py-2 pr-4 font-semibold">{heading}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {card.risk_conditions.map((condition) => {
                    const entry = pooled[condition];
                    const interval = entry?.auc_ci;
                    return (
                      <tr key={condition}>
                        <td className="py-2.5 pr-4 font-semibold">{condition}</td>
                        <td className="py-2.5 pr-4 tabular-nums">
                          {entry?.auc?.toFixed(3) ?? "—"}
                        </td>
                        <td className="py-2.5 pr-4 tabular-nums text-slate-600">
                          {interval ? `[${interval.ci_low.toFixed(2)}, ${interval.ci_high.toFixed(2)}]` : "—"}
                        </td>
                        <td className="py-2.5 pr-4 text-xs">
                          {card.confound_disclosure.severity_by_condition[condition] ?? "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-4 text-xs leading-5 text-slate-500">
              Confidence intervals are bootstrap percentile intervals over held-out subjects.
              At this cohort size the interval, not the point estimate, is the result.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center gap-3">
            <Ban className="size-5 text-red-700" />
            <h2 className="section-title">Out of scope</h2>
          </CardHeader>
          <CardContent>
            <p className="mb-3 text-sm leading-6 text-slate-600">
              {String(card.intended_use.purpose ?? "")}
            </p>
            <ul className="space-y-2 text-sm leading-6 text-slate-600">
              {outOfScope.map((item) => (
                <li key={item} className="flex gap-3">
                  <Ban className="mt-1 size-4 shrink-0 text-red-700" />
                  {item}
                </li>
              ))}
            </ul>
            <p className="mt-4 rounded-md bg-slate-50 p-3 text-xs leading-5 text-slate-600">
              {String(card.intended_use.disclaimer ?? "")}
            </p>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
