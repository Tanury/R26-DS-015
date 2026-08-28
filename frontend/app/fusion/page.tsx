"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Info,
  Mic,
  Scale,
  ShieldAlert,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import { PageHeader } from "@/components/page-header";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  buildFusionResult,
  fusionCandidates,
  fusionConditions,
  type FusionModality,
} from "@/lib/fusion";
import { useStoredHistory } from "@/lib/history";
import type { AssessmentType, HistoryItem } from "@/lib/types";
import { percent } from "@/lib/utils";

const modalities: {
  type: FusionModality;
  label: string;
  href: string;
  icon: typeof Mic;
  color: string;
}[] = [
  { type: "Voice", label: "Voice", href: "/voice", icon: Mic, color: "text-blue-700 bg-blue-50" },
  { type: "General", label: "Biomedical", href: "/general", icon: ClipboardList, color: "text-emerald-700 bg-emerald-50" },
  { type: "EEG", label: "EEG", href: "/eeg", icon: Activity, color: "text-violet-700 bg-violet-50" },
];

const conditionNames = {
  AD: "Alzheimer’s disease pattern",
  PD: "Parkinson’s disease pattern",
  MS: "Multiple sclerosis pattern",
};

function dateLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown date" : date.toLocaleString();
}

function levelTone(level: string): string {
  if (level === "High") return "bg-red-100 text-red-800";
  if (level === "Moderate") return "bg-amber-100 text-amber-800";
  return "bg-emerald-100 text-emerald-800";
}

export default function FusionResultPage() {
  const items = useStoredHistory();
  const candidates = useMemo(() => Object.fromEntries(
    modalities.map(({ type }) => [type, fusionCandidates(items, type as AssessmentType)]),
  ) as Record<FusionModality, HistoryItem[]>, [items]);
  const [selectedIds, setSelectedIds] = useState<Record<FusionModality, string>>({
    Voice: "",
    General: "",
    EEG: "",
  });

  const effectiveSelectedIds = useMemo(() => Object.fromEntries(
    modalities.map(({ type }) => [
      type,
      candidates[type].some((item) => item.id === selectedIds[type])
        ? selectedIds[type]
        : candidates[type][0]?.id ?? "",
    ]),
  ) as Record<FusionModality, string>, [candidates, selectedIds]);

  const selected = useMemo(() => Object.fromEntries(
    modalities.flatMap(({ type }) => {
      const match = candidates[type].find((item) => item.id === effectiveSelectedIds[type]);
      return match ? [[type, match]] : [];
    }),
  ) as Partial<Record<FusionModality, HistoryItem>>, [candidates, effectiveSelectedIds]);
  const ready = modalities.every(({ type }) => Boolean(selected[type]));
  const result = useMemo(() => {
    if (!ready) return null;
    return buildFusionResult(selected as Record<FusionModality, HistoryItem>);
  }, [ready, selected]);

  return (
    <AppShell>
      <BackButton href="/" label="Back to Assessments" />
      <PageHeader
        title="Multimodal Fusion Result"
        description="Combine one Voice, one Biomedical, and one EEG assessment into a transparent, quality-adjusted neurological risk summary."
      />

      <div className="mb-6 flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
        <ShieldAlert className="mt-0.5 size-5 shrink-0" />
        <div>
          <div className="font-semibold">Confirm the assessments belong to the same participant</div>
          <p className="mt-1">The browser history has no shared patient identifier. The latest result of each type is selected automatically; change a selection below if the dates or participant do not match.</p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <h2 className="section-title">Component Selection</h2>
          <p className="mt-1 text-sm leading-6 text-slate-500">All three independently completed assessments are required. Fusion is recalculated immediately when a selection changes.</p>
        </CardHeader>
        <CardContent className="grid gap-4 lg:grid-cols-3">
          {modalities.map(({ type, label, href, icon: Icon, color }) => {
            const available = candidates[type];
            return (
              <div key={type} className="rounded-lg border border-slate-200 p-4">
                <div className="flex items-center gap-3">
                  <div className={`grid size-10 place-items-center rounded-lg ${color}`}><Icon className="size-5" /></div>
                  <div>
                    <div className="font-semibold">{label} Assessment</div>
                    <div className="text-xs text-slate-500">{available.length} available</div>
                  </div>
                  {available.length > 0 && <CheckCircle2 className="ml-auto size-5 text-emerald-700" />}
                </div>
                {available.length ? (
                  <select
                    className="mt-4 h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm"
                    aria-label={`Select ${label} assessment`}
                    value={effectiveSelectedIds[type]}
                    onChange={(event) => setSelectedIds((current) => ({ ...current, [type]: event.target.value }))}
                  >
                    {available.map((item) => <option key={item.id} value={item.id}>{item.id} · {dateLabel(item.createdAt)}</option>)}
                  </select>
                ) : (
                  <Button className="mt-4 w-full" variant="outline" asChild>
                    <Link href={href}>Complete {label} assessment <ArrowRight className="size-4" /></Link>
                  </Button>
                )}
              </div>
            );
          })}
        </CardContent>
      </Card>

      {!result ? (
        <Card className="mt-6 border-dashed">
          <CardContent className="py-14 text-center">
            <BrainCircuit className="mx-auto size-12 text-slate-400" />
            <h2 className="mt-4 text-xl font-bold">Fusion result is not ready</h2>
            <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-500">Complete the missing assessment components. The combined result will appear here automatically when Voice, Biomedical, and EEG outputs are available.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="mt-6 space-y-6">
          <Card className="border-blue-200">
            <CardContent className="flex flex-col justify-between gap-6 p-6 sm:flex-row sm:items-center sm:p-8">
              <div className="flex items-start gap-4">
                {result.riskLevel === "Low" ? <CheckCircle2 className="mt-1 size-10 shrink-0 text-emerald-700" /> : <AlertTriangle className="mt-1 size-10 shrink-0 text-amber-700" />}
                <div>
                  <div className="text-xs font-bold uppercase tracking-wide text-blue-700">Quality-adjusted fusion output</div>
                  <h1 className="mt-2 text-3xl font-bold sm:text-4xl">{result.headline}</h1>
                  <p className="mt-3 max-w-3xl leading-7 text-slate-600">{result.explanation}</p>
                </div>
              </div>
              <div className="grid min-w-44 grid-cols-2 gap-3 text-center sm:grid-cols-1">
                <div className="rounded-lg bg-slate-50 px-5 py-3"><div className="text-xs font-bold uppercase text-slate-500">Highest score</div><div className="mt-1 text-3xl font-bold text-blue-700">{Math.round(result.highestScore * 100)}<span className="text-base text-slate-500"> / 100</span></div></div>
                <div className="rounded-lg bg-slate-50 px-5 py-3"><div className="text-xs font-bold uppercase text-slate-500">Agreement</div><div className="mt-1 text-xl font-bold">{result.agreement}</div><div className="text-xs text-slate-500">{result.agreeingModalities}/3 rank {result.highestCondition} first</div></div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-5 lg:grid-cols-3">
            {fusionConditions.map((condition) => {
              const fused = result.conditions[condition];
              return (
                <Card key={condition} className={condition === result.highestCondition ? "border-blue-300 ring-1 ring-blue-100" : ""}>
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between gap-3">
                      <div><div className="text-2xl font-bold">{condition}</div><div className="mt-1 text-xs text-slate-500">{conditionNames[condition]}</div></div>
                      <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${levelTone(fused.level)}`}>{fused.level}</span>
                    </div>
                    <div className="mt-5 flex items-end justify-between"><span className="text-sm font-semibold text-slate-500">Fused score</span><span className="text-3xl font-bold">{Math.round(fused.score * 100)}</span></div>
                    <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-200"><div className={`h-full rounded-full ${fused.level === "High" ? "bg-red-700" : fused.level === "Moderate" ? "bg-amber-600" : "bg-emerald-700"}`} style={{ width: percent(fused.score) }} /></div>
                    <div className="mt-5 space-y-3 border-t border-slate-200 pt-4">
                      {fused.contributions.map((contribution) => <div key={contribution.modality}><div className="flex justify-between text-sm"><span className="font-semibold">{contribution.modality === "General" ? "Biomedical" : contribution.modality}</span><span>{percent(contribution.evidence)}</span></div><div className="mt-1 text-xs text-slate-500">Effective weight {Math.round(contribution.effectiveWeight * 100)}% · {contribution.qualityLabel}</div></div>)}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3"><Scale className="size-6 text-blue-700" /><h2 className="section-title">How the Fusion Was Calculated</h2></div>
                <div className="mt-4 space-y-4 text-sm leading-6 text-slate-600">
                  <p>Voice and Biomedical disease probabilities are each compared directly with their Healthy probability: disease ÷ (disease + Healthy). This produces one-vs-Healthy evidence for AD, PD, and MS.</p>
                  <p>For each condition, those two values are averaged with the matching independent EEG score. The mean is weighted by saved voice extraction quality, biomedical input completeness, EEG signal quality, and the EEG model’s condition-specific confound severity.</p>
                  <div className="rounded-lg border border-blue-100 bg-blue-50 p-4 text-blue-950"><div className="font-semibold">Fusion formula</div><p className="mt-1 font-mono text-xs">Σ(component evidence × effective weight) ÷ Σ(effective weights)</p></div>
                  <p>Low is below 40, Moderate is 40–74, and High is 75 or above. These are transparent decision-support bands, not clinically calibrated thresholds.</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h2 className="section-title">Suggested Actions</h2>
                <ol className="mt-4 space-y-4 text-sm leading-6 text-slate-600">
                  {result.suggestions.map((suggestion, index) => <li key={suggestion} className="flex gap-3"><span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-bold text-emerald-800">{index + 1}</span><span>{suggestion}</span></li>)}
                </ol>
              </CardContent>
            </Card>
          </div>

          <Card className="border-amber-200 bg-amber-50">
            <CardContent className="p-6">
              <div className="flex gap-3"><Info className="mt-0.5 size-5 shrink-0 text-amber-800" /><div><h2 className="font-bold text-amber-950">Limitations that travel with this result</h2><ul className="mt-3 space-y-2 text-sm leading-6 text-amber-950">{result.cautions.map((caution) => <li key={caution}>• {caution}</li>)}</ul></div></div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h2 className="section-title">Source Assessments</h2>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">{modalities.map(({ type, label }) => { const item = selected[type]!; return <div key={type} className="rounded-lg bg-slate-50 p-4"><div className="text-xs font-bold uppercase text-slate-500">{label}</div><div className="mt-1 font-mono text-sm font-semibold">{result.sourceIds[type]}</div><div className="mt-1 text-xs text-slate-500">{dateLabel(item.createdAt)}</div></div>; })}</div>
            </CardContent>
          </Card>

          <div className="flex gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-900"><AlertTriangle className="mt-0.5 size-5 shrink-0" /><div><div className="font-semibold">Urgent symptoms override every model result</div><p className="mt-1">Seek emergency care for sudden weakness or numbness, confusion, trouble speaking, vision or balance problems, or a sudden severe headache. Do not wait for another assessment or fusion result.</p></div></div>
          <ResearchDisclaimer />
        </div>
      )}
    </AppShell>
  );
}
