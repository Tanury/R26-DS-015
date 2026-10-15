"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import { EegReportView } from "@/components/eeg-report-view";
import { Button } from "@/components/ui/button";
import { fetchCohortReport } from "@/lib/eeg-api";
import type { EegRiskReport } from "@/lib/eeg-types";

export default function CohortSamplePage() {
  const { subjectId } = useParams<{ subjectId: string }>();
  return <SampleReport key={subjectId} subjectId={subjectId} />;
}

function SampleReport({ subjectId }: { subjectId: string }) {
  const [report, setReport] = useState<EegRiskReport | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let cancelled = false;
    fetchCohortReport(subjectId).then((value) => {
      if (!cancelled) setReport(value);
    }).catch((reason: unknown) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to open this sample.");
    });
    return () => { cancelled = true; };
  }, [subjectId, attempt]);

  return <AppShell>
    <BackButton href="/eeg/prediction" label="Return to your saved EEG prediction" />
    <p className="mb-5 text-sm text-slate-600">Reference sample report. Your saved assessment remains available.</p>
    {report ? <EegReportView report={report} /> : error ? (
      <div role="alert" className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900">
        <p>{error}</p>
        <Button variant="outline" className="mt-3" onClick={() => { setError(""); setAttempt((value) => value + 1); }}>Try again</Button>
      </div>
    ) : <p role="status" className="py-10 text-center text-slate-500">Loading EEG sample report…</p>}
  </AppShell>;
}
