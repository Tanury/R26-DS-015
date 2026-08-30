"use client";

import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Info,
  RefreshCw,
  ShieldAlert,
  Waves,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import { EegPredictionScores } from "@/components/eeg-prediction-scores";
import { EegRecommendations } from "@/components/eeg-recommendations";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { buildEegUserGuidance } from "@/lib/eeg-prediction-guidance";
import { useCurrentResult } from "@/lib/history";

export default function EegPredictionPage() {
  const item = useCurrentResult();
  if (!item || item.type !== "EEG" || !item.eegReport) {
    return (
      <AppShell>
        <BackButton href="/eeg" label="Back to EEG Assessment" />
        <div className="mx-auto max-w-xl py-20 text-center">
          <BrainCircuit className="mx-auto size-12 text-blue-700" />
          <h1 className="mt-5 text-2xl font-bold">No current EEG prediction</h1>
          <p className="mt-2 leading-7 text-slate-600">Upload an EEG recording or open a cohort result before viewing the prediction and recommendations.</p>
          <Button className="mt-6" asChild><Link href="/eeg">Start EEG assessment</Link></Button>
        </div>
      </AppShell>
    );
  }

  const report = item.eegReport;
  const guidance = buildEegUserGuidance(report);
  const favorable = guidance.top.condition === "Healthy";

  return (
    <AppShell>
      <BackButton href="/eeg/results" label="Back to Detailed EEG Result" />
      <div className="space-y-6">
        <Card className="border-blue-200">
          <CardContent className="flex flex-col justify-between gap-6 p-6 sm:flex-row sm:items-center sm:p-8">
            <div className="flex items-start gap-4">
              {favorable
                ? <CheckCircle2 className="mt-1 size-10 shrink-0 text-emerald-700" />
                : <AlertTriangle className="mt-1 size-10 shrink-0 text-amber-700" />}
              <div>
                <div className="text-xs font-bold uppercase tracking-wide text-blue-700">EEG prediction and user guidance</div>
                <h1 className="mt-2 text-3xl font-bold sm:text-4xl">{guidance.headline}</h1>
                <p className="mt-3 max-w-3xl leading-7 text-slate-600">{guidance.summary}</p>
                <p className="mt-2 text-xs text-slate-500">Subject {report.subject_id} · {report.source === "cohort" ? "precomputed cohort report" : "uploaded EEG recording"}</p>
              </div>
            </div>
            <div className="min-w-44 rounded-lg border border-slate-200 bg-slate-50 px-6 py-4 text-center">
              <div className="mt-1 text-4xl font-bold text-blue-700">{guidance.top.condition}</div>
              <div className="mt-1 text-sm font-semibold">{guidance.top.points} / 100 points</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h2 className="section-title">Condition Predictions</h2>
            <p className="mt-1 text-sm leading-6 text-slate-500">All four condition patterns are shown, with the selected result emphasized by its tinted card.</p>
          </CardHeader>
          <CardContent>
            <EegPredictionScores conditions={guidance.conditions} />
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <EegRecommendations recommendations={guidance.recommendations} />

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-3"><Waves className="size-6 text-blue-700" /><h2 className="section-title">Result Reliability</h2></div>
              <div className="mt-5 rounded-lg bg-slate-50 p-4 text-sm leading-6 text-slate-700">{guidance.signalQualitySummary}</div>
              <ul className="mt-5 space-y-3 text-sm leading-6 text-slate-600">{guidance.cautions.map((caution) => <li key={caution} className="flex gap-3"><Info className="mt-1 size-4 shrink-0 text-blue-700" /><span>{caution}</span></li>)}</ul>
            </CardContent>
          </Card>
        </div>

        <div className="flex gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-900">
          <ShieldAlert className="mt-0.5 size-5 shrink-0" />
          <div><div className="font-semibold">Sudden neurological symptoms require urgent help</div><p className="mt-1">Seek emergency care immediately for sudden weakness or numbness, confusion, trouble speaking, new vision or balance problems, seizure, loss of consciousness, or a sudden severe headache. Do not wait for or repeat this assessment.</p></div>
        </div>

        <div className="flex flex-wrap gap-3">
          <Button asChild><Link href="/eeg/results"><Activity className="size-4" />View Detailed EEG Result</Link></Button>
          <Button variant="outline" asChild><Link href="/fusion"><BrainCircuit className="size-4" />Fusion Result</Link></Button>
          <Button variant="outline" asChild><Link href="/eeg"><RefreshCw className="size-4" />New EEG Assessment</Link></Button>
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs leading-6 text-slate-600">{report.clinical_disclaimer}</div>
      </div>
    </AppShell>
  );
}
