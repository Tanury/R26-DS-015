"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { fetchSimilarEegSamples } from "@/lib/eeg-api";
import { buildEegUserGuidance } from "@/lib/eeg-prediction-guidance";
import type { EegRiskReport, SimilarEegResponse } from "@/lib/eeg-types";

const BANDS = ["delta", "theta", "alpha", "beta", "low_gamma"];
const LABELS = ["Delta", "Theta", "Alpha", "Beta", "Low gamma"];

export function EegDecisionEvidence({ report }: { report: EegRiskReport }) {
  const guidance = buildEegUserGuidance(report);
  return (
    <Card>
      <CardHeader>
        <h2 className="section-title">Why this result over the alternatives?</h2>
        <p className="mt-1 text-sm text-slate-500">Measured scores, supporting observations, and limits of the evidence.</p>
      </CardHeader>
      <CardContent className="space-y-5">
        <ol className="list-decimal space-y-3 pl-5 text-sm leading-6 text-slate-700">
          {guidance.evidence.map((evidence) => <li key={evidence}>{evidence}</li>)}
        </ol>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-left text-sm">
            <caption className="sr-only">Comparison of all EEG class scores and reasons for ranking</caption>
            <thead><tr className="border-b text-xs text-slate-500"><th scope="col" className="py-3 pr-4">Pattern</th><th scope="col" className="py-3 pr-4">{guidance.scoreLabel}</th><th scope="col" className="py-3">Why it ranks here</th></tr></thead>
            <tbody>{guidance.conditions.map((condition) => (
              <tr key={condition.condition} className={condition.isPrimary ? "border-b bg-blue-50" : "border-b border-slate-100"}>
                <th scope="row" className="p-3 pl-2">{condition.condition}</th>
                <td className="p-3 tabular-nums">{condition.points === null ? "Not available" : `${condition.points}%`}</td>
                <td className="p-3 text-xs leading-6 text-slate-600">{condition.explanation}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <details className="rounded-lg border border-amber-200 bg-amber-50 p-4" open={report.confound_disclosure.has_critical}>
          <summary className="cursor-pointer text-sm font-semibold text-amber-950">Interpretation limits and model confounds</summary>
          <ul className="mt-3 list-disc space-y-2 pl-4 text-xs leading-6 text-amber-950">
            {guidance.cautions.map((caution) => <li key={caution}>{caution}</li>)}
          </ul>
        </details>
      </CardContent>
    </Card>
  );
}

export function EegWaveform({ report }: { report: EegRiskReport }) {
  const preview = report.waveform_preview;
  const channels = Object.entries(preview?.channels ?? {}).filter(([, samples]) => samples.length > 1 && samples.every(Number.isFinite));
  const available = !!preview && preview.sampling_rate_hz > 0 && channels.length > 0;
  const amplitude = Math.max(1, ...channels.flatMap(([, values]) => values.map(Math.abs)));
  return (
    <Card>
      <CardHeader><h2 className="section-title">Your EEG signal sample</h2><p className="mt-1 text-sm text-slate-500">A measured excerpt from the assessed recording.</p></CardHeader>
      <CardContent>
        {!available ? <p className="rounded-lg bg-slate-50 p-4 text-sm leading-6 text-slate-600">This saved report does not contain waveform samples. Upload the original EEG again to view a measured excerpt. Its band-power profile can still be compared below.</p> : (
          <>
            <div className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-3">
              <svg viewBox={`0 0 900 ${channels.length * 85 + 35}`} className="min-w-[600px] w-full" role="img" aria-label={`Measured EEG waveform for ${channels.map(([name]) => name).join(", ")}, in microvolts`}>
                {channels.map(([name, samples], index) => {
                  const baseline = index * 85 + 42;
                  const duration = (samples.length - 1) / preview.sampling_rate_hz;
                  return <g key={name}>
                    <text x="0" y={baseline + 4} fontSize="13" fill="#475569">{name}</text>
                    <line x1="50" x2="865" y1={baseline} y2={baseline} stroke="#cbd5e1" />
                    {[0, 1, 2, 3, 4].map((tick) => <line key={tick} x1={50 + tick * 203.75} x2={50 + tick * 203.75} y1={baseline - 38} y2={baseline + 38} stroke="#e2e8f0" />)}
                    <polyline fill="none" stroke="#1d4ed8" strokeWidth="1.2" points={samples.map((value, i) => `${50 + i / (samples.length - 1) * 815},${baseline - value / amplitude * 34}`).join(" ")} />
                    {index === channels.length - 1 && [0, 1, 2, 3, 4].map((tick) => <text key={tick} x={50 + tick * 203.75} y={baseline + 63} textAnchor="middle" fontSize="12" fill="#64748b">{(tick * duration / 4).toFixed(2)} s</text>)}
                  </g>;
                })}
              </svg>
            </div>
            <p className="mt-3 text-xs leading-6 text-slate-500">Shared vertical scale: ±{amplitude.toFixed(1)} µV around each electrode baseline · {preview.sampling_rate_hz} Hz. {preview.note}</p>
            {report.signal_quality.source_kind === "pre_epoched" && <p className="mt-2 text-xs text-amber-800">This recording contains reassembled epochs; joins may not be physiologically continuous.</p>}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function EegSimilarSamples({ report }: { report: EegRiskReport }) {
  // Remount on report changes so a preceding subject's comparison cannot flash.
  return <SimilarSamples key={`${report.subject_id}|${report.generated_at}`} report={report} />;
}

function SimilarSamples({ report }: { report: EegRiskReport }) {
  const [result, setResult] = useState<SimilarEegResponse | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [selectedId, setSelectedId] = useState("");
  const complete = !!report.model_summary.run_id && BANDS.every((band) => Number.isFinite(report.band_power_profile[band]));
  useEffect(() => {
    if (!complete) return;
    const controller = new AbortController();
    fetchSimilarEegSamples(report, controller.signal).then((value) => {
      if (!controller.signal.aborted) setResult(value);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Unable to load similar EEG samples.");
    });
    return () => controller.abort();
  }, [report, complete, attempt]);
  const selected = result?.samples.find((sample) => sample.subject_id === selectedId) ?? result?.samples[0];
  const guidance = buildEegUserGuidance(report);
  const closest = result?.samples[0];
  const disagree = closest && guidance.top.isPrimary && (closest.true_class === "HC" ? "Healthy" : closest.true_class) !== guidance.top.condition;
  return (
    <Card>
      <CardHeader><h2 className="section-title">Similar EEG samples from the cohort</h2><p className="mt-1 text-sm leading-6 text-slate-500">Compare your measured frequency profile with real examples from every available group.</p></CardHeader>
      <CardContent>
        {!complete ? <p className="text-sm text-slate-600">This report is missing the band measurements or model version needed for a compatible comparison.</p>
          : error ? <div role="alert" className="rounded-lg bg-amber-50 p-4 text-sm text-amber-900"><p>Sample comparison unavailable: {error}</p><Button variant="outline" size="sm" className="mt-3" onClick={() => { setError(""); setAttempt((value) => value + 1); }}>Retry comparison</Button></div>
            : !result ? <p role="status" className="py-4 text-sm text-slate-500">Finding similar measured profiles…</p>
              : !selected ? <p className="text-sm text-slate-600">No other compatible cohort recordings are available for this model version.</p> : (
                <div className="space-y-5">
                  <div className="rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm leading-6 text-blue-950">
                    Closest band-power example: <strong>{closest?.subject_id}</strong> · cohort label <strong>{closest?.true_class}</strong>.
                    {disagree && <p className="mt-2 font-semibold">This example has a different label from the model’s selected pattern. Band-power similarity does not independently support that selection.</p>}
                    <p className="mt-1 text-xs">Compared {result.compared_subjects} recordings. A nearest example always exists when candidates exist; proximity alone does not establish a good match.</p>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    {result.samples.map((sample) => (
                      <button key={sample.subject_id} type="button" aria-pressed={sample.subject_id === selected.subject_id} onClick={() => setSelectedId(sample.subject_id)} className={`min-w-0 rounded-lg border p-4 text-left transition-colors ${sample.subject_id === selected.subject_id ? "border-blue-600 bg-blue-50" : "border-slate-200 hover:bg-slate-50"}`}>
                        <span className="block text-lg font-bold">{sample.true_class === "HC" ? "Healthy control" : sample.true_class}</span>
                        <span className="mt-1 block break-all font-mono text-xs text-slate-500">{sample.subject_id}</span>
                        <span className="mt-3 block text-sm font-semibold">{(sample.distance * 100).toFixed(2)} pp difference</span>
                        <span className="text-xs text-slate-500">Site {sample.site} · {sample.signal_quality} quality</span>
                      </button>
                    ))}
                  </div>
                  <div className="flex flex-wrap gap-5 text-xs font-semibold"><span className="text-blue-700">■ Your recording</span><span className="text-violet-700">■ {selected.subject_id}</span></div>
                  <div className="space-y-4">
                    {BANDS.map((band, index) => <div key={band}>
                      <div className="mb-1 flex justify-between text-xs"><span className="font-semibold">{LABELS[index]}</span><span className="tabular-nums text-slate-600">{(report.band_power_profile[band] * 100).toFixed(1)}% / {(selected.band_power_profile[band] * 100).toFixed(1)}%</span></div>
                      <div className="space-y-1" role="img" aria-label={`${LABELS[index]}: your recording ${(report.band_power_profile[band] * 100).toFixed(1)} percent, sample ${(selected.band_power_profile[band] * 100).toFixed(1)} percent`}>
                        <div className="h-2 rounded bg-slate-100"><div className="h-2 rounded bg-blue-700" style={{ width: `${report.band_power_profile[band] * 100}%` }} /></div>
                        <div className="h-2 rounded bg-slate-100"><div className="h-2 rounded bg-violet-600" style={{ width: `${selected.band_power_profile[band] * 100}%` }} /></div>
                      </div>
                    </div>)}
                  </div>
                  <p className="text-xs leading-6 text-slate-500">Shared scale: 0–100% relative band power. Difference is expressed in percentage points (pp), not a disease probability. {result.method} Age and recording site are not matched.</p>
                  <Button variant="outline" size="sm" asChild><Link href={`/eeg/cohort/${encodeURIComponent(selected.subject_id)}`}>Open this sample’s full report</Link></Button>
                </div>
              )}
      </CardContent>
    </Card>
  );
}
