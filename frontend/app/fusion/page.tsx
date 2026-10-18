"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Calculator,
  CheckCircle2,
  ClipboardList,
  Info,
  Mic,
  RefreshCw,
  Scale,
  ScanSearch,
  ShieldAlert,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import {
  BiomedicalUploadPanel,
  EegUploadPanel,
  ImagingUploadPanel,
  VoiceUploadPanel,
} from "@/components/fusion-uploads";
import { PageHeader } from "@/components/page-header";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  buildFusionResult,
  fusionCandidates,
  fusionConditions,
  imagingCandidates,
  type CoreFusionModality,
  type CoreSelection,
  type FusionCondition,
  type FusionResult,
  type ImagingSelection,
} from "@/lib/fusion";
import { useStoredHistory } from "@/lib/history";
import { fetchImagingCohort, type ImagingCohort } from "@/lib/neuroimaging-api";
import type { AssessmentType, HistoryItem } from "@/lib/types";
import { percent } from "@/lib/utils";

/** Where a component's result comes from, or whether it is left out. */
type Source = "saved" | "upload" | "none";

const modalities: {
  type: CoreFusionModality;
  label: string;
  icon: typeof Mic;
  color: string;
}[] = [
  { type: "Voice", label: "Voice", icon: Mic, color: "text-blue-700 bg-blue-50" },
  { type: "General", label: "Biomedical", icon: ClipboardList, color: "text-emerald-700 bg-emerald-50" },
  { type: "EEG", label: "EEG", icon: Activity, color: "text-violet-700 bg-violet-50" },
];

const imagingScanNames: Record<FusionCondition, string> = {
  AD: "AD structural MRI",
  PD: "PD DaTscan / MRI",
  MS: "MS OCT / MRI",
};

const contributionLabel: Record<string, string> = {
  General: "Biomedical",
};

const modalityLabel: Record<string, string> = {
  Voice: "Voice",
  General: "Biomedical",
  EEG: "EEG",
  Neuroimaging: "Neuroimaging",
};

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

function SourceToggle({
  value,
  onChange,
  savedCount,
  label,
}: {
  value: Source;
  onChange: (source: Source) => void;
  savedCount: number;
  label: string;
}) {
  const options: { source: Source; text: string; disabled?: boolean }[] = [
    { source: "saved", text: `Saved (${savedCount})`, disabled: savedCount === 0 },
    { source: "upload", text: "Upload" },
    { source: "none", text: "Not included" },
  ];
  return (
    <div className="mt-4 grid grid-cols-3 gap-1 rounded-lg bg-slate-100 p-1" role="radiogroup" aria-label={`${label} source`}>
      {options.map(({ source, text, disabled }) => (
        <button
          key={source}
          type="button"
          role="radio"
          aria-checked={value === source}
          disabled={disabled}
          onClick={() => onChange(source)}
          className={`rounded-md px-2 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
            value === source ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"
          }`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

/** A one-line reading of whatever result a component slot currently holds. */
function ResultSummary({ item, uploaded }: { item: HistoryItem; uploaded?: boolean }) {
  let headline = "";
  let detail = "";
  if (item.eegReport) {
    const assessment = item.eegReport.risk_assessment;
    const top = assessment.conditions[assessment.highest_risk_condition];
    headline = `Highest EEG risk ${assessment.highest_risk_condition}`;
    detail = top ? `${percent(top.risk_score)} · ${top.risk_band} · ${item.eegReport.signal_quality.grade} signal` : "";
  } else if (item.neuroimaging) {
    const scan = item.neuroimaging;
    headline = `Predicted ${scan.prediction}`;
    detail = `${scan.confidence.toFixed(1)}% confidence · ${scan.riskLevel} risk · ${scan.modality}${scan.filename ? ` · ${scan.filename}` : ""}`;
  } else if (item.prediction) {
    headline = `Predicted ${item.prediction.predicted_class}`;
    detail = `${percent(item.prediction.confidence_score)} confidence · ${item.prediction.risk_level} risk`;
  }
  const scan = item.neuroimaging;
  return (
    <div className="mt-2 rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-600">
      <span className="font-semibold text-slate-900">{headline}</span>
      {uploaded && <span className="ml-1 rounded bg-sky-100 px-1.5 py-0.5 font-semibold text-sky-800">Uploaded for fusion</span>}
      {scan?.demo && <span className="ml-1 rounded bg-amber-100 px-1.5 py-0.5 font-semibold text-amber-800">Demo output</span>}
      <div className="mt-0.5 text-slate-500">{detail}</div>
      {scan?.cohort && (
        <div className="mt-0.5 text-slate-500">
          True label {scan.cohort.trueLabel} · {scan.cohort.embeddingDim}-d z_img · model LOOCV {Math.round(scan.cohort.loocvAccuracy * 100)}% acc, AUC {scan.cohort.loocvAuc.toFixed(2)}
        </div>
      )}
    </div>
  );
}

function SlotPlaceholder({ children }: { children: string }) {
  return <div className="mt-2 rounded-md border border-dashed border-slate-300 px-3 py-3 text-xs text-slate-500">{children}</div>;
}

export default function FusionResultPage() {
  const items = useStoredHistory();
  const candidates = useMemo(() => Object.fromEntries(
    modalities.map(({ type }) => [type, fusionCandidates(items, type as AssessmentType)]),
  ) as Record<CoreFusionModality, HistoryItem[]>, [items]);

  const [cohort, setCohort] = useState<ImagingCohort | null>(null);
  const [cohortError, setCohortError] = useState<string | null>(null);
  useEffect(() => {
    fetchImagingCohort()
      .then(setCohort)
      .catch((error: unknown) => setCohortError(error instanceof Error ? error.message : "Could not load the imaging cohort."));
  }, []);
  // Saved scans first, then the neuroimaging module's processed cohort.
  const scans = useMemo(() => {
    const pool = [...items, ...(cohort?.items ?? [])];
    return Object.fromEntries(
      fusionConditions.map((condition) => [condition, imagingCandidates(pool, condition)]),
    ) as Record<FusionCondition, HistoryItem[]>;
  }, [items, cohort]);

  // null = default: a component with saved results starts on "saved", otherwise "not included".
  const [coreSource, setCoreSource] = useState<Record<CoreFusionModality, Source | null>>({ Voice: null, General: null, EEG: null });
  const [coreIds, setCoreIds] = useState<Record<CoreFusionModality, string>>({ Voice: "", General: "", EEG: "" });
  const [coreUploads, setCoreUploads] = useState<Partial<Record<CoreFusionModality, HistoryItem>>>({});
  // Imaging stays out of fusion until a scan is chosen or uploaded.
  const [imagingSource, setImagingSource] = useState<Record<FusionCondition, Source>>({ AD: "none", PD: "none", MS: "none" });
  const [imagingIds, setImagingIds] = useState<Record<FusionCondition, string>>({ AD: "", PD: "", MS: "" });
  const [imagingUploads, setImagingUploads] = useState<Partial<Record<FusionCondition, HistoryItem>>>({});

  const effectiveCoreSource = (type: CoreFusionModality): Source =>
    coreSource[type] ?? (candidates[type].length ? "saved" : "none");

  const selection = useMemo(() => {
    const core: CoreSelection = {};
    for (const { type } of modalities) {
      const source = coreSource[type] ?? (candidates[type].length ? "saved" : "none");
      if (source === "saved") core[type] = candidates[type].find((item) => item.id === coreIds[type]) ?? candidates[type][0];
      if (source === "upload" && coreUploads[type]) core[type] = coreUploads[type];
    }
    const imaging: ImagingSelection = {};
    for (const condition of fusionConditions) {
      const source = imagingSource[condition];
      if (source === "saved") imaging[condition] = scans[condition].find((item) => item.id === imagingIds[condition]) ?? scans[condition][0];
      if (source === "upload" && imagingUploads[condition]) imaging[condition] = imagingUploads[condition];
    }
    return { core, imaging };
  }, [candidates, coreSource, coreIds, coreUploads, scans, imagingSource, imagingIds, imagingUploads]);

  const includedNames = [
    ...modalities.filter(({ type }) => selection.core[type]).map(({ label }) => label),
    ...fusionConditions.filter((condition) => selection.imaging[condition]).map((condition) => `Neuroimaging (${condition})`),
  ];
  const selectionKey = JSON.stringify([
    modalities.map(({ type }) => selection.core[type]?.id ?? null),
    fusionConditions.map((condition) => selection.imaging[condition]?.id ?? null),
  ]);

  const [computed, setComputed] = useState<{ result: FusionResult; key: string; core: CoreSelection; imaging: ImagingSelection } | null>(null);
  const [calcError, setCalcError] = useState<string | null>(null);
  const stale = computed !== null && computed.key !== selectionKey;
  const result = computed?.result ?? null;

  function calculate() {
    setCalcError(null);
    try {
      setComputed({
        result: buildFusionResult(selection.core, selection.imaging),
        key: selectionKey,
        core: selection.core,
        imaging: selection.imaging,
      });
    } catch (error) {
      setComputed(null);
      setCalcError(error instanceof Error ? error.message : "The fusion result could not be calculated.");
    }
  }

  return (
    <AppShell>
      <BackButton href="/" label="Back to Assessments" />
      <PageHeader
        title="Multimodal Fusion Result"
        description="Fuse whichever Voice, Biomedical, EEG, and Neuroimaging results are available into one quality-adjusted classification: AD, PD, MS, or Healthy control (HC)."
      />

      <div className="mb-6 flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
        <ShieldAlert className="mt-0.5 size-5 shrink-0" />
        <div>
          <div className="font-semibold">Confirm the assessments belong to the same participant</div>
          <p className="mt-1">Saved results have no shared patient identifier. Check that every saved result you include belongs to this participant and a comparable time period.</p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <h2 className="section-title">Component Selection</h2>
          <p className="mt-1 text-sm leading-6 text-slate-500">
            For each component, use a saved result, upload data here, or mark it Not included. Uploads on this page are used for this fusion only and are not saved as individual assessments. Excluded components are left out of the calculation.
          </p>
        </CardHeader>
        <CardContent className="grid gap-4 lg:grid-cols-3">
          {modalities.map(({ type, label, icon: Icon, color }) => {
            const available = candidates[type];
            const source = effectiveCoreSource(type);
            const item = selection.core[type];
            return (
              <div key={type} className={`rounded-lg border p-4 ${source === "none" ? "border-dashed border-slate-200 bg-slate-50/50" : "border-slate-200"}`}>
                <div className="flex items-center gap-3">
                  <div className={`grid size-10 place-items-center rounded-lg ${color}`}><Icon className="size-5" /></div>
                  <div>
                    <div className="font-semibold">{label} Assessment</div>
                    <div className="text-xs text-slate-500">{source === "none" ? "Not included" : item ? "Included" : "Waiting for upload"}</div>
                  </div>
                  {item && <CheckCircle2 className="ml-auto size-5 text-emerald-700" />}
                </div>
                <SourceToggle
                  label={label}
                  value={source}
                  savedCount={available.length}
                  onChange={(next) => setCoreSource((current) => ({ ...current, [type]: next }))}
                />
                {source === "saved" && (
                  <select
                    className="mt-3 h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm"
                    aria-label={`Select ${label} assessment`}
                    value={item?.id ?? ""}
                    onChange={(event) => setCoreIds((current) => ({ ...current, [type]: event.target.value }))}
                  >
                    {available.map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.id} · {dateLabel(candidate.createdAt)}</option>)}
                  </select>
                )}
                {source === "upload" && type === "Voice" && <VoiceUploadPanel onResult={(result) => setCoreUploads((current) => ({ ...current, Voice: result }))} />}
                {source === "upload" && type === "General" && <BiomedicalUploadPanel onResult={(result) => setCoreUploads((current) => ({ ...current, General: result }))} />}
                {source === "upload" && type === "EEG" && <EegUploadPanel onResult={(result) => setCoreUploads((current) => ({ ...current, EEG: result }))} />}
                {source === "none" && <SlotPlaceholder>{`${label} is left out of the fusion calculation.`}</SlotPlaceholder>}
                {item && source !== "none" && <ResultSummary item={item} uploaded={source === "upload"} />}
              </div>
            );
          })}

          <div className="rounded-lg border border-slate-200 p-4 lg:col-span-3">
            <div className="flex items-center gap-3">
              <div className="grid size-10 place-items-center rounded-lg bg-sky-50 text-sky-700"><ScanSearch className="size-5" /></div>
              <div>
                <div className="font-semibold">Neuroimaging Assessment</div>
                <div className="text-xs text-slate-500">Each imaging model scores one condition, so each scan informs only that condition&apos;s fused score. Saved results include the neuroimaging module&apos;s processed MS cohort.</div>
              </div>
              <span className="ml-auto text-xs text-slate-500">{Object.keys(selection.imaging).length}/3 conditions included</span>
              {Object.keys(selection.imaging).length > 0 && <CheckCircle2 className="size-5 text-emerald-700" />}
            </div>
            {cohortError && <div className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900">Processed imaging cohort unavailable: {cohortError}</div>}
            <div className="mt-2 grid gap-4 sm:grid-cols-3">
              {fusionConditions.map((condition) => {
                const available = scans[condition];
                const savedScans = available.filter((item) => !item.neuroimaging?.cohort);
                const processed = available.filter((item) => item.neuroimaging?.cohort);
                const source = imagingSource[condition];
                const item = selection.imaging[condition];
                return (
                  <div key={condition}>
                    <SourceToggle
                      label={`${condition} neuroimaging`}
                      value={source}
                      savedCount={available.length}
                      onChange={(next) => setImagingSource((current) => ({ ...current, [condition]: next }))}
                    />
                    <div className="mt-2 text-xs font-bold uppercase text-slate-500">{imagingScanNames[condition]}</div>
                    {source === "saved" && (
                      <select
                        className="mt-2 h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm"
                        aria-label={`Select ${condition} neuroimaging result`}
                        value={item?.id ?? ""}
                        onChange={(event) => setImagingIds((current) => ({ ...current, [condition]: event.target.value }))}
                      >
                        {savedScans.length > 0 && <optgroup label="Saved scans">{savedScans.map((scan) => <option key={scan.id} value={scan.id}>{scan.id} · {scan.neuroimaging?.scanType}{scan.neuroimaging?.demo ? " (demo)" : ""} · {dateLabel(scan.createdAt)}</option>)}</optgroup>}
                        {processed.length > 0 && <optgroup label="Processed cohort">{processed.map((scan) => <option key={scan.id} value={scan.id}>{scan.id} · {scan.neuroimaging?.scanType} · true {scan.neuroimaging?.cohort?.trueLabel}</option>)}</optgroup>}
                      </select>
                    )}
                    {source === "upload" && (
                      <ImagingUploadPanel condition={condition} onResult={(result) => setImagingUploads((current) => ({ ...current, [condition]: result }))} />
                    )}
                    {source === "none" && (
                      <SlotPlaceholder>
                        {available.length === 0 && cohort?.conditions_missing[condition]
                          ? `Not included. ${cohort.conditions_missing[condition]} Choose Upload to analyze a local scan.`
                          : `${condition} imaging is left out of the fusion calculation.`}
                      </SlotPlaceholder>
                    )}
                    {item && source !== "none" && <ResultSummary item={item} uploaded={source === "upload"} />}
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardContent className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm">
            <div className="font-semibold">
              {includedNames.length ? `Included: ${includedNames.join(", ")}` : "No components included yet"}
            </div>
            <div className="mt-0.5 text-slate-500">
              {includedNames.length
                ? "Components marked Not included are left out of the calculation."
                : "Select a saved result or upload data for at least one component."}
            </div>
          </div>
          <Button type="button" size="lg" disabled={includedNames.length === 0} onClick={calculate}>
            {computed && stale ? <RefreshCw className="size-4" /> : <Calculator className="size-4" />}
            {computed && stale ? "Recalculate fusion result" : "Calculate fusion result"}
          </Button>
        </CardContent>
      </Card>
      {calcError && <div className="mt-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">{calcError}</div>}
      {stale && (
        <div className="mt-3 flex gap-2 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          The selection changed after this result was calculated. Press Recalculate to update it.
        </div>
      )}

      {!result ? (
        <Card className="mt-6 border-dashed">
          <CardContent className="py-14 text-center">
            <BrainCircuit className="mx-auto size-12 text-slate-400" />
            <h2 className="mt-4 text-xl font-bold">No fusion result yet</h2>
            <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-500">Choose a saved result, upload data, or mark each component Not included, then press Calculate fusion result.</p>
          </CardContent>
        </Card>
      ) : (
        <div className={`mt-6 space-y-6 ${stale ? "opacity-60" : ""}`}>
          <Card className="border-blue-200">
            <CardContent className="flex flex-col justify-between gap-6 p-6 sm:flex-row sm:items-center sm:p-8">
              <div className="flex items-start gap-4">
                {result.finalClass === "HC" ? <CheckCircle2 className="mt-1 size-10 shrink-0 text-emerald-700" /> : <AlertTriangle className="mt-1 size-10 shrink-0 text-amber-700" />}
                <div>
                  <div className="text-xs font-bold uppercase tracking-wide text-blue-700">
                    {result.includedModalities.length}-modality fusion · {result.includedModalities.map((modality) => modalityLabel[modality]).join(" + ")}
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <span className={`rounded-lg px-4 py-2 text-4xl font-black sm:text-5xl ${result.finalClass === "HC" ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-800"}`}>{result.finalClass}</span>
                    <h1 className="text-2xl font-bold sm:text-3xl">{result.headline}</h1>
                  </div>
                  <p className="mt-3 max-w-3xl leading-7 text-slate-600">{result.explanation}</p>
                  {result.excludedModalities.length > 0 && (
                    <p className="mt-2 text-sm text-slate-500">Not included: {result.excludedModalities.map((modality) => modalityLabel[modality]).join(", ")}</p>
                  )}
                </div>
              </div>
              <div className="grid min-w-44 grid-cols-2 gap-3 text-center sm:grid-cols-1">
                <div className="rounded-lg bg-slate-50 px-5 py-3"><div className="text-xs font-bold uppercase text-slate-500">Highest score</div><div className="mt-1 text-3xl font-bold text-blue-700">{Math.round(result.highestScore * 100)}<span className="text-base text-slate-500"> / 100</span></div></div>
                <div className="rounded-lg bg-slate-50 px-5 py-3">
                  <div className="text-xs font-bold uppercase text-slate-500">Agreement</div>
                  <div className="mt-1 text-xl font-bold">{result.agreement}</div>
                  {result.rankingModalities > 1 && <div className="text-xs text-slate-500">{result.agreeingModalities}/{result.rankingModalities} rank {result.highestCondition} first</div>}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-5 lg:grid-cols-3">
            {fusionConditions.map((condition) => {
              const fused = result.conditions[condition];
              return (
                <Card key={condition} className={condition === result.highestCondition ? "border-blue-300 ring-1 ring-blue-100" : fused.informed ? "" : "border-dashed"}>
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between gap-3">
                      <div><div className="text-2xl font-bold">{condition}</div><div className="mt-1 text-xs text-slate-500">{conditionNames[condition]}</div></div>
                      {fused.informed
                        ? <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${levelTone(fused.level)}`}>{fused.level}</span>
                        : <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-500">Not scored</span>}
                    </div>
                    {fused.informed ? (
                      <>
                        <div className="mt-5 flex items-end justify-between"><span className="text-sm font-semibold text-slate-500">Fused score</span><span className="text-3xl font-bold">{Math.round(fused.score * 100)}</span></div>
                        <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-200"><div className={`h-full rounded-full ${fused.level === "High" ? "bg-red-700" : fused.level === "Moderate" ? "bg-amber-600" : "bg-emerald-700"}`} style={{ width: percent(fused.score) }} /></div>
                        <div className="mt-5 space-y-3 border-t border-slate-200 pt-4">
                          {fused.contributions.map((contribution) => <div key={contribution.modality}><div className="flex justify-between text-sm"><span className="font-semibold">{contributionLabel[contribution.modality] ?? contribution.modality}</span><span>{percent(contribution.evidence)}</span></div><div className="mt-1 text-xs text-slate-500">Effective weight {Math.round(contribution.effectiveWeight * 100)}% · {contribution.qualityLabel}</div></div>)}
                        </div>
                      </>
                    ) : (
                      <p className="mt-5 text-sm leading-6 text-slate-500">No included component carries evidence for {condition}, so it was not scored.</p>
                    )}
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
                  <p>Only included components take part. Voice and Biomedical disease probabilities are each compared directly with their Healthy probability: disease ÷ (disease + Healthy). This produces one-vs-Healthy evidence for AD, PD, and MS.</p>
                  <p>For each condition, the included evidence is averaged with the matching independent EEG score when EEG is included. The mean is weighted by voice extraction quality, biomedical input completeness, EEG signal quality, and the EEG model’s condition-specific confound severity.</p>
                  <p>A neuroimaging result is added only to its own condition: PD and MS scans use disease ÷ (disease + Healthy Control); the AD MRI uses (AD + ½·MCI) ÷ (AD + MCI + CN). Agreement is counted across the included models that score every condition.</p>
                  <div className="rounded-lg border border-blue-100 bg-blue-50 p-4 text-blue-950"><div className="font-semibold">Fusion formula</div><p className="mt-1 font-mono text-xs">Σ(component evidence × effective weight) ÷ Σ(effective weights)</p></div>
                  <p>Final classification: the condition with the highest fused score is reported when it reaches 50/100 (disease evidence outweighs healthy evidence); otherwise the result is Healthy control (HC).</p>
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

          {computed && (
            <Card>
              <CardContent className="p-6">
                <h2 className="section-title">Source Assessments</h2>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  {modalities.filter(({ type }) => computed.core[type]).map(({ type, label }) => {
                    const item = computed.core[type]!;
                    return <div key={type} className="rounded-lg bg-slate-50 p-4"><div className="text-xs font-bold uppercase text-slate-500">{label}</div><div className="mt-1 font-mono text-sm font-semibold">{item.id}</div><div className="mt-1 text-xs text-slate-500">{item.id.startsWith("FUS-") ? "Uploaded for fusion" : dateLabel(item.createdAt)}</div></div>;
                  })}
                  {result.imagingConditions.map((condition) => {
                    const item = computed.imaging[condition]!;
                    return <div key={condition} className="rounded-lg bg-slate-50 p-4"><div className="text-xs font-bold uppercase text-slate-500">Neuroimaging · {condition}</div><div className="mt-1 font-mono text-sm font-semibold">{item.id}</div><div className="mt-1 text-xs text-slate-500">{item.neuroimaging?.scanType} · {item.id.startsWith("FUS-") ? "Uploaded for fusion" : dateLabel(item.createdAt)}</div></div>;
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          <div className="flex gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-900"><AlertTriangle className="mt-0.5 size-5 shrink-0" /><div><div className="font-semibold">Urgent symptoms override every model result</div><p className="mt-1">Seek emergency care for sudden weakness or numbness, confusion, trouble speaking, vision or balance problems, or a sudden severe headache. Do not wait for another assessment or fusion result.</p></div></div>
          <ResearchDisclaimer />
        </div>
      )}
    </AppShell>
  );
}
