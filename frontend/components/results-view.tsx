"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, ClipboardPlus, Gauge, Info, RefreshCw, Waves } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { ProbabilityBars } from "@/components/probability-bars";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
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

  return (
    <AppShell>
      <div className="space-y-6">
        <Card className="p-1">
          <CardContent className="flex flex-col justify-between gap-5 p-6 sm:flex-row sm:items-center sm:p-8">
            <div className="flex items-start gap-4">
              {healthy ? <CheckCircle2 className="mt-1 size-10 shrink-0 text-emerald-700" /> : <AlertTriangle className="mt-1 size-10 shrink-0 text-amber-700" />}
              <div><h1 className="text-3xl font-bold sm:text-5xl">{healthy ? "Healthy Pattern" : `${prediction.predicted_class} Risk Pattern`}</h1><p className="mt-2 max-w-3xl text-base leading-7 text-slate-600">{prediction.observed_issues[0]}</p></div>
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

        <div className="grid gap-6 lg:grid-cols-2">
          <Card><CardContent className="p-6"><h2 className="section-title">Observed Patterns</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600">{prediction.observed_issues.map((issue) => <li key={issue} className="flex gap-3"><Info className="mt-0.5 size-4 shrink-0 text-blue-700" />{issue}</li>)}</ul></CardContent></Card>
          <Card><CardContent className="p-6"><h2 className="section-title">Next Steps</h2><ul className="mt-4 space-y-3 text-sm leading-6 text-slate-600">{prediction.recommendations.map((item) => <li key={item} className="flex gap-3"><CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-700" />{item}</li>)}</ul></CardContent></Card>
        </div>

        {item.transcript && <Card><CardContent className="p-6"><h2 className="section-title">Speech Transcript</h2><p className="mt-3 text-sm italic leading-7 text-slate-600">“{item.transcript}”</p></CardContent></Card>}

        <div className="flex flex-wrap gap-3"><Button asChild><Link href={repeatHref}><RefreshCw className="size-4" />New assessment</Link></Button><Button variant="outline" asChild><Link href="/history">View history</Link></Button></div>
        <ResearchDisclaimer />
      </div>
    </AppShell>
  );
}
