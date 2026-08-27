"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, ClipboardPlus, ExternalLink, Gauge, Info, RefreshCw, ShieldAlert, Waves } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { ProbabilityBars } from "@/components/probability-bars";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  biomarkerDetails,
  biomarkerGroups,
  buildGeneralInterpretation,
  formatBiomarkerValue,
} from "@/lib/general-assessment";
import type { HistoryItem } from "@/lib/types";
import { percent } from "@/lib/utils";

export function ResultsView({ item, expectedType }: { item: HistoryItem | null; expectedType: "Voice" | "General" }) {
  if (!item || item.type !== expectedType || !item.prediction) {
    const href = expectedType === "Voice" ? "/voice" : "/general";
    return (
      <AppShell><div className="mx-auto max-w-xl py-20 text-center"><ClipboardPlus className="mx-auto size-12 text-blue-700" /><h1 className="mt-5 text-2xl font-bold">No current result</h1><p className="mt-2 text-slate-600">Complete an assessment to view its result.</p><Button className="mt-6" asChild><Link href={href}>Start assessment</Link></Button></div></AppShell>
    );
  }

  const { prediction } = item;
  const healthy = prediction.predicted_class === "Healthy";
  const risk = Math.round(prediction.risk_score * 100);
  const repeatHref = expectedType === "Voice" ? "/voice" : "/general";
  const generalInterpretation = expectedType === "General"
    ? buildGeneralInterpretation(prediction)
    : null;
  const favorableResult = generalInterpretation
    ? generalInterpretation.lowerRiskHealthy
    : healthy;
  const resultHeadline = generalInterpretation?.headline
    ?? (healthy ? "Healthy Pattern" : `${prediction.predicted_class} Risk Pattern`);
  const resultSummary = generalInterpretation?.resultMeaning
    ?? prediction.observed_issues[0];

  return (
    <AppShell>
      <div className="space-y-6">
        <Card className="p-1">
          <CardContent className="flex flex-col justify-between gap-5 p-6 sm:flex-row sm:items-center sm:p-8">
            <div className="flex items-start gap-4">
              {favorableResult ? <CheckCircle2 className="mt-1 size-10 shrink-0 text-emerald-700" /> : <AlertTriangle className="mt-1 size-10 shrink-0 text-amber-700" />}
              <div><div className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">{expectedType === "General" ? "General biomedical screening result" : "Voice assessment result"}</div><h1 className="text-3xl font-bold sm:text-5xl">{resultHeadline}</h1><p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">{resultSummary}</p></div>
            </div>
            <div className="shrink-0 rounded-lg border border-slate-300 bg-slate-50 px-7 py-4 text-center"><div className="text-xs font-bold uppercase text-slate-500">Confidence</div><div className="mt-1 text-4xl font-bold text-blue-700">{percent(prediction.confidence_score)}</div></div>
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card><CardHeader><h2 className="section-title">Probability Distribution</h2></CardHeader><CardContent><ProbabilityBars probabilities={prediction.probabilities} /></CardContent></Card>
          <Card><CardHeader><h2 className="section-title">Risk Summary</h2></CardHeader><CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg bg-slate-50 p-5"><Gauge className="size-6 text-blue-700" /><div className="mt-4 text-sm font-semibold text-slate-500">Overall risk score</div><div className="mt-1 text-4xl font-bold">{risk}<span className="text-lg text-slate-500"> / 100</span></div></div>
            <div className="rounded-lg bg-slate-50 p-5"><Waves className="size-6 text-emerald-700" /><div className="mt-4 text-sm font-semibold text-slate-500">Risk level</div><div className="mt-1 text-3xl font-bold capitalize">{prediction.risk_level}</div></div>
          </CardContent></Card>
        </div>

        {generalInterpretation ? (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card><CardContent className="p-6"><h2 className="section-title">How to Read This Result</h2><div className="mt-4 space-y-4 text-sm leading-6 text-slate-600"><div className="rounded-lg border border-blue-100 bg-blue-50 p-4"><div className="font-semibold text-slate-900">What the classifier selected</div><p className="mt-1">{generalInterpretation.resultMeaning}</p></div><div><div className="font-semibold text-slate-900">How the risk score was calculated</div><p className="mt-1">{generalInterpretation.formula}</p></div><div><div className="font-semibold text-slate-900">How distinct the result was</div><p className="mt-1">{generalInterpretation.separation}</p></div><div><div className="font-semibold text-slate-900">What the risk band means</div><p className="mt-1">{generalInterpretation.bandExplanation}</p></div><div><div className="font-semibold text-slate-900">Model-reported context</div><ul className="mt-2 space-y-2">{prediction.observed_issues.map((issue) => <li key={issue} className="flex gap-2"><Info className="mt-1 size-3.5 shrink-0 text-blue-700" /><span>{issue}</span></li>)}</ul></div></div></CardContent></Card>
              <Card><CardContent className="p-6"><h2 className="section-title">Suggested Actions</h2><p className="mt-2 text-sm leading-6 text-slate-500">Prioritized from the selected risk class. Use laboratory quality, symptoms, and clinical evaluation—not this score alone—to make healthcare decisions.</p><ol className="mt-4 space-y-3 text-sm leading-6 text-slate-600">{generalInterpretation.suggestions.map((suggestion, index) => <li key={suggestion} className="flex gap-3"><span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-bold text-emerald-800">{index + 1}</span><span>{suggestion}</span></li>)}</ol></CardContent></Card>
            </div>

            {item.biomarkers && <Card><CardContent className="p-6"><div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start"><div><h2 className="section-title">Submitted Model Inputs</h2><p className="mt-2 max-w-4xl text-sm leading-6 text-slate-500">These are the exact 24 fields sent to both fitted pipelines. “Not supplied” values were imputed from training-set statistics. The model does not provide per-feature attribution, so no individual value can honestly be called the cause of the result. Clinical instrument, laboratory method, specimen handling, units, and collection timing can affect comparability.</p></div><span className="w-fit shrink-0 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">24 inputs reviewed</span></div><div className="mt-6 grid gap-4 lg:grid-cols-2 xl:grid-cols-4">{biomarkerGroups.map((group) => <div key={group.title} className="rounded-lg border border-slate-200 p-4"><h3 className="font-semibold text-slate-900">{group.title}</h3><p className="mt-1 min-h-12 text-xs leading-5 text-slate-500">{group.description}</p><dl className="mt-4 space-y-3">{group.keys.map((key) => <div key={key} className="border-t border-slate-100 pt-3"><div className="flex items-start justify-between gap-3"><dt className="text-sm font-medium text-slate-700">{biomarkerDetails[key].label}</dt><dd className="shrink-0 text-right text-sm font-bold text-slate-900">{formatBiomarkerValue(key, item.biomarkers!)}</dd></div><p className="mt-1 text-xs leading-5 text-slate-500">{biomarkerDetails[key].description}</p></div>)}</dl></div>)}</div></CardContent></Card>}

            <div className="flex gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-900"><ShieldAlert className="mt-0.5 size-5 shrink-0" /><div><div className="font-semibold">Sudden neurological symptoms need urgent help</div><p className="mt-1">Seek emergency care immediately for sudden face, arm, or leg weakness or numbness; new confusion or trouble speaking; sudden vision or balance problems; or a sudden severe headache. Do not wait for or repeat this screening.</p><a className="mt-2 inline-flex items-center gap-2 font-semibold underline" href="https://www.cdc.gov/stroke/signs-symptoms/index.html" target="_blank" rel="noreferrer">CDC stroke warning signs<ExternalLink className="size-4" /></a></div></div>
          </>
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card><CardContent className="p-6"><h2 className="section-title">Observed Patterns</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600">{prediction.observed_issues.map((issue) => <li key={issue} className="flex gap-3"><Info className="mt-0.5 size-4 shrink-0 text-blue-700" />{issue}</li>)}</ul></CardContent></Card>
            <Card><CardContent className="p-6"><h2 className="section-title">Next Steps</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600">{prediction.recommendations.map((recommendation) => <li key={recommendation} className="flex gap-3"><CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-700" />{recommendation}</li>)}</ul></CardContent></Card>
          </div>
        )}

        {item.transcript && <Card><CardContent className="p-6"><h2 className="section-title">Speech Transcript</h2><p className="mt-3 text-sm italic leading-7 text-slate-600">“{item.transcript}”</p></CardContent></Card>}

        <div className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-700"><Info className="mt-0.5 size-5 shrink-0 text-blue-700" /><div><div className="font-semibold text-slate-900">Research-use limitation</div><p className="mt-1">{prediction.disclaimer}</p></div></div>

        <div className="flex flex-wrap gap-3"><Button asChild><Link href={repeatHref}><RefreshCw className="size-4" />New assessment</Link></Button><Button variant="outline" asChild><Link href="/history">View history</Link></Button></div>
        <ResearchDisclaimer />
      </div>
    </AppShell>
  );
}
