"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useMemo, useState } from "react";
import { AudioLines, CheckCircle2, Info, Loader2, Timer } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { submitGeneralAssessment } from "@/lib/api";
import { saveAssessment } from "@/lib/history";
import { featureKeys, type FeatureKey, type SpeechFeatures } from "@/lib/types";

const defaults: SpeechFeatures = { mfcc_1_mean: -245.2, mfcc_2_mean: 82.1, mfcc_3_mean: 14.7, pitch_mean: 178.4, pitch_std: 32.8, jitter: 0.9, shimmer: 3.1, hnr: 18.6, speech_rate: 2.4, pause_count: 12, mean_pause_duration: 0.42, mean_energy: 61.3, spectral_centroid_mean: 1840.5, zero_crossing_rate_mean: 0.08 };
const labels: Record<FeatureKey, [string, string]> = { mfcc_1_mean:["MFCC 1 Mean", ""], mfcc_2_mean:["MFCC 2 Mean", ""], mfcc_3_mean:["MFCC 3 Mean", ""], pitch_mean:["Pitch Mean", "Hz"], pitch_std:["Pitch SD", "Hz"], jitter:["Jitter", "%"], shimmer:["Shimmer", "%"], hnr:["HNR", "dB"], speech_rate:["Speech Rate", "words/s"], pause_count:["Pause Count", ""], mean_pause_duration:["Mean Pause Duration", "sec"], mean_energy:["Mean Energy", "dB"], spectral_centroid_mean:["Spectral Centroid", "Hz"], zero_crossing_rate_mean:["Zero Crossing Rate", "ratio"] };

export default function GeneralAssessmentPage() {
  const router = useRouter();
  const [values, setValues] = useState<Record<FeatureKey, string>>(Object.fromEntries(featureKeys.map((key) => [key, String(defaults[key])])) as Record<FeatureKey,string>);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const validCount = useMemo(() => featureKeys.filter((key) => values[key] !== "" && Number.isFinite(Number(values[key]))).length, [values]);

  async function submit(event: FormEvent) {
    event.preventDefault(); setError("");
    if (validCount !== featureKeys.length) { setError("Enter a finite numeric value for every feature."); return; }
    const features = Object.fromEntries(featureKeys.map((key) => [key, Number(values[key])])) as SpeechFeatures;
    setLoading(true);
    try { const prediction = await submitGeneralAssessment(features); saveAssessment("General", prediction, { features }); router.push("/general/results"); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Assessment failed."); }
    finally { setLoading(false); }
  }

  const groups: [string, typeof AudioLines, FeatureKey[]][] = [
    ["Acoustic Characteristics", AudioLines, ["mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","pitch_mean","pitch_std","jitter","shimmer","hnr"]],
    ["Timing and Spectral Features", Timer, ["speech_rate","pause_count","mean_pause_duration","mean_energy","spectral_centroid_mean","zero_crossing_rate_mean"]],
  ];

  return <AppShell><PageHeader title="General Speech Risk Assessment" description="Evaluate neurological speech-risk patterns using the exact 14 acoustic and timing features expected by the classifier." />
    <form onSubmit={submit} className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-6">{groups.map(([title, Icon, keys]) => <Card key={title}><CardContent className="p-6"><div className="mb-6 flex items-center justify-between gap-3"><h2 className="section-title flex items-center gap-2"><Icon className="size-6 text-blue-700" />{title}</h2><span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">{keys.length} features</span></div><div className="grid gap-5 md:grid-cols-2">{keys.map((key) => <label key={key} className="block"><span className="mb-2 flex items-center gap-2 text-sm font-semibold">{labels[key][0]}<Info className="size-3.5 text-slate-400" /></span><div className="relative"><Input type="number" step="any" value={values[key]} onChange={(event) => setValues((current) => ({...current,[key]:event.target.value}))} required /><span className="pointer-events-none absolute right-3 top-3 text-xs text-slate-400">{labels[key][1]}</span></div></label>)}</div></CardContent></Card>)}</div>
      <Card className="xl:sticky xl:top-24"><CardContent className="p-6"><h2 className="section-title">Input Validation</h2><div className="mt-5 flex gap-3 rounded-lg bg-slate-50 p-4"><CheckCircle2 className={validCount === 14 ? "size-6 shrink-0 text-emerald-700" : "size-6 shrink-0 text-amber-700"}/><div><div className="font-semibold">{validCount} / 14 features available</div><p className="mt-1 text-sm leading-6 text-slate-500">Every value is validated again by FastAPI before model scaling.</p></div></div><div className="mt-6 h-2 overflow-hidden rounded-full bg-slate-200"><div className="h-full bg-emerald-700 transition-all" style={{width:`${validCount/14*100}%`}} /></div>{error && <div role="alert" className="mt-5 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">{error}</div>}<Button type="submit" size="lg" className="mt-6 w-full" disabled={loading || validCount !== 14}>{loading ? <><Loader2 className="size-4 animate-spin" />Analyzing</> : "Run Risk Assessment"}</Button></CardContent></Card>
    </form>
  </AppShell>;
}
