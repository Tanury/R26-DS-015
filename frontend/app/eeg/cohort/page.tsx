"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Search } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import { EmbeddingScatter } from "@/components/embedding-scatter";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { fetchCohort, fetchCohortReport, fetchProjection } from "@/lib/eeg-api";
import type { CohortPage, CohortProjection } from "@/lib/eeg-types";
import { isFlaggedSeverity } from "@/lib/eeg-types";
import { saveEegAssessment } from "@/lib/history";
import { percent } from "@/lib/utils";

const CLASS_CHIP: Record<string, string> = {
  HC: "bg-teal-100 text-teal-800",
  AD: "bg-red-100 text-red-800",
  PD: "bg-amber-100 text-amber-800",
  MS: "bg-blue-100 text-blue-800",
};

const CONDITION_NAMES: Record<string, string> = {
  HC: "healthy control",
  AD: "alzheimer alzheimers disease",
  PD: "parkinson parkinsons disease",
  MS: "multiple sclerosis",
};

function matchesSearch(subject: CohortPage["subjects"][number], query: string) {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return true;

  const riskValues = Object.entries(subject.risk_scores).flatMap(([condition, score]) => [
    condition,
    CONDITION_NAMES[condition] ?? "",
    String(score),
    String(Math.round(score * 100)),
  ]);
  const searchable = [
    "subject id filename",
    subject.subject_id,
    "class condition diagnosis",
    subject.true_class,
    CONDITION_NAMES[subject.true_class] ?? "",
    "site",
    subject.site,
    "source type",
    subject.source_kind,
    "quality",
    subject.signal_quality,
    "age",
    subject.age === null ? "age unavailable" : String(subject.age),
    "top highest risk",
    subject.highest_risk_condition,
    CONDITION_NAMES[subject.highest_risk_condition] ?? "",
    "confound status",
    subject.confound_severity,
    ...riskValues,
  ].join(" ").toLowerCase();

  return terms.every((term) => searchable.includes(term));
}

export default function EegCohortPage() {
  const router = useRouter();
  // Local state rather than useSearchParams: that hook forces the tree up to the
  // nearest Suspense boundary to client-render, and the existing history page
  // already filters this way.
  const [trueClass, setTrueClass] = useState("");
  const [site, setSite] = useState("");
  const [quality, setQuality] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState<CohortPage | null>(null);
  const [projection, setProjection] = useState<CohortProjection | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [opening, setOpening] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await fetchCohort({
          trueClass: trueClass || undefined,
          site: site || undefined,
          quality: quality || undefined,
          limit: 200,
        });
        if (!cancelled) setPage(result);
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to load cohort.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [trueClass, site, quality]);

  useEffect(() => {
    let cancelled = false;
    fetchProjection()
      .then((result) => {
        if (!cancelled) setProjection(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = useMemo(() => {
    const subjects = page?.subjects ?? [];
    return subjects.filter((subject) => matchesSearch(subject, search));
  }, [page, search]);

  async function open(subjectId: string) {
    setOpening(true);
    setError("");
    try {
      const report = await fetchCohortReport(subjectId);
      saveEegAssessment(report);
      router.push("/eeg/results");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to open that subject.");
      setOpening(false);
    }
  }

  const filters = page?.available_filters ?? {};

  return (
    <AppShell>
      <BackButton href="/eeg" label="Back to EEG Assessment" />
      <PageHeader
        title="EEG Cohort Explorer"
        description="Every recording assessed by the encoder. Filter, inspect the embedding space, and open any subject's full report."
      />

      {error && (
        <div role="alert" className="mb-6 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <Card>
        <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-3 size-5 text-slate-400" />
            <Input
              className="pl-10"
              placeholder="Search ID, condition, site, quality, age, or risk"
              aria-label="Search all cohort subject information"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          {[
            { label: "All classes", value: trueClass, set: setTrueClass, options: filters.true_class ?? [] },
            { label: "All sites", value: site, set: setSite, options: filters.site ?? [] },
            { label: "All quality", value: quality, set: setQuality, options: filters.signal_quality ?? [] },
          ].map((control) => (
            <select
              key={control.label}
              className="h-11 rounded-md border border-slate-300 bg-white px-3 text-sm"
              value={control.value}
              onChange={(event) => control.set(event.target.value)}
            >
              <option value="">{control.label}</option>
              {control.options.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          ))}
        </CardContent>
      </Card>

      {projection && (
        <Card className="mt-6">
          <CardHeader className="flex flex-row items-center justify-between gap-3">
            <h2 className="section-title">z_eeg Embedding Space</h2>
            {/* Coverage belongs next to the plot, not only in the caption below it —
                a partial projection looks exactly like a complete one. */}
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                page && projection.points.length < page.total
                  ? "bg-amber-100 text-amber-800"
                  : "bg-slate-100 text-slate-600"
              }`}
            >
              {projection.points.length}
              {page ? ` of ${page.total}` : ""} subjects plotted
            </span>
          </CardHeader>
          <CardContent>
            <EmbeddingScatter
              points={projection.points}
              selected={selected}
              onSelect={(id) => setSelected(id)}
              note={projection.note}
              explainedVariance={projection.explained_variance}
              agreement={projection.neighbourhood_agreement}
            />
            {selected && (
              <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg bg-slate-50 p-3">
                <span className="font-mono text-sm font-semibold">{selected}</span>
                <Button size="sm" disabled={opening} onClick={() => open(selected)}>
                  Open report
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setSelected(null)}>
                  Clear
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card className="mt-6 overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between">
          <h2 className="section-title">
            Subjects{page ? ` (${rows.length} of ${page.total})` : ""}
          </h2>
          {opening && <Loader2 className="size-5 animate-spin text-blue-700" />}
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-left text-sm">
            <thead className="bg-slate-100 text-xs uppercase text-slate-500">
              <tr>
                {["Subject", "Class", "Site", "Quality", "Epochs", "AD", "PD", "MS", "Top risk", ""].map(
                  (heading, index) => (
                    <th key={`${heading}-${index}`} className="px-4 py-3 font-semibold">
                      {heading}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {rows.map((subject) => (
                <tr key={subject.subject_id} className="hover:bg-slate-50">
                  <td className="px-4 py-3 font-mono text-xs">{subject.subject_id}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                        CLASS_CHIP[subject.true_class] ?? "bg-slate-100 text-slate-700"
                      }`}
                    >
                      {subject.true_class}
                    </span>
                  </td>
                  <td className="px-4 py-3">{subject.site}</td>
                  <td className="px-4 py-3">{subject.signal_quality}</td>
                  <td className="px-4 py-3 tabular-nums">{subject.epochs_used}</td>
                  {["AD", "PD", "MS"].map((condition) => (
                    <td key={condition} className="px-4 py-3 tabular-nums">
                      {percent(subject.risk_scores[condition] ?? 0)}
                    </td>
                  ))}
                  <td className="px-4 py-3">
                    <span className="font-semibold">{subject.highest_risk_condition}</span>
                    {isFlaggedSeverity(subject.confound_severity) && (
                      <span
                        className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-800"
                        title={subject.confound_severity}
                      >
                        confounded
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={opening}
                      onClick={() => open(subject.subject_id)}
                    >
                      View
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {page && rows.length === 0 && (
            <div className="p-12 text-center text-sm text-slate-500">
              No subjects match the search and selected filters.
            </div>
          )}
          {!page && !error && (
            <div className="p-12 text-center text-sm text-slate-500">Loading cohort…</div>
          )}
        </div>
      </Card>
    </AppShell>
  );
}
