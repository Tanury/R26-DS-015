"use client";

import Link from "next/link";
import { BrainCircuit } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { EegReportView } from "@/components/eeg-report-view";
import { Button } from "@/components/ui/button";
import { useCurrentResult } from "@/lib/history";

export default function EegResultsPage() {
  const item = useCurrentResult();

  if (!item || item.type !== "EEG" || !item.eegReport) {
    return (
      <AppShell>
        <div className="mx-auto max-w-xl py-20 text-center">
          <BrainCircuit className="mx-auto size-12 text-blue-700" />
          <h1 className="mt-5 text-2xl font-bold">No current EEG result</h1>
          <p className="mt-2 text-slate-600">
            Open a cohort subject or upload a recording to view an assessment.
          </p>
          <Button className="mt-6" asChild>
            <Link href="/eeg">Start EEG assessment</Link>
          </Button>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <EegReportView report={item.eegReport} />
    </AppShell>
  );
}
