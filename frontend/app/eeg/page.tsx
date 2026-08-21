"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  BrainCircuit,
  FileUp,
  Loader2,
  ServerCrash,
  UploadCloud,
  Users,
  X,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  fetchCohort,
  fetchCohortReport,
  fetchModelCard,
  pollEegJob,
  submitEegAssessment,
} from "@/lib/eeg-api";
import type {
  CohortSubject,
  EegJob,
  EegModelCard,
  EegRejectionDetails,
} from "@/lib/eeg-types";
import { saveEegAssessment } from "@/lib/history";
import { percent } from "@/lib/utils";

const MAX_BYTES = 120 * 1024 * 1024;

export default function EegAssessmentPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [card, setCard] = useState<EegModelCard | null>(null);
  const [featured, setFeatured] = useState<CohortSubject[]>([]);
  const [loadError, setLoadError] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<EegJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [rejection, setRejection] = useState<EegRejectionDetails | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [modelCard, cohort] = await Promise.all([
          fetchModelCard(),
          fetchCohort({ limit: 6 }),
        ]);
        if (cancelled) return;
        setCard(modelCard);
        setFeatured(cohort.subjects);
      } catch (reason) {
        if (!cancelled) {
          setLoadError(reason instanceof Error ? reason.message : "Unable to reach the EEG API.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function acceptFiles(candidates: FileList | null) {
    setError("");
    if (!candidates?.length) return;
    const chosen = Array.from(candidates).slice(0, 2);
    const bad = chosen.find((file) => !/\.(set|fdt)$/i.test(file.name));
    if (bad) {
      setError(`'${bad.name}' is not an EEGLAB file. Choose a .set file and its .fdt companion.`);
      return;
    }
    const total = chosen.reduce((sum, file) => sum + file.size, 0);
    if (total > MAX_BYTES) {
      setError("The upload exceeds the 120 MB limit.");
      return;
    }
    if (!chosen.some((file) => file.name.toLowerCase().endsWith(".set"))) {
      setError("A .set file is required.");
      return;
    }
    setFiles(chosen);
  }

  async function analyze() {
    setError("");
    setRejection(null);
    setBusy(true);
    try {
      const created = await submitEegAssessment(files);
      setJob(created);
      const settled = await pollEegJob(created.job_id, setJob);
      if (settled.status === "completed" && settled.report) {
        saveEegAssessment(settled.report);
        router.push("/eeg/results");
        return;
      }
      setError(settled.error?.message ?? "The assessment could not be completed.");
      // A rejected recording gets a structured explanation: which electrodes were
      // too noisy and by how much. Far more actionable than "it failed".
      if (settled.error?.code === "insufficient_quality") {
        setRejection(settled.error.details as EegRejectionDetails);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "EEG assessment failed.");
    } finally {
      setBusy(false);
    }
  }

  async function openCohortSubject(subjectId: string) {
    setError("");
    setBusy(true);
    try {
      const report = await fetchCohortReport(subjectId);
      saveEegAssessment(report);
      router.push("/eeg/results");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load that subject.");
      setBusy(false);
    }
  }

  const inferenceOff = card !== null && !card.inference_available;

  return (
    <AppShell>
      <PageHeader
        title="EEG Neurological Risk Assessment"
        description="Explore assessed cohort recordings, or upload an EEGLAB recording for a full pipeline run producing three independent risk scores and a 256-D z_eeg embedding."
      />

      {loadError && (
        <div role="alert" className="mb-6 flex gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          <ServerCrash className="mt-0.5 size-5 shrink-0" />
          <div>
            <p className="font-semibold">The EEG API is not reachable.</p>
            <p className="mt-1">{loadError}</p>
            <p className="mt-1 text-xs">
              Start the backend, then run{" "}
              <code className="rounded bg-red-100 px-1">
                python scripts/build_eeg_cohort_index.py --fixtures
              </code>
              .
            </p>
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center gap-3">
            <div className="grid size-11 place-items-center rounded-lg bg-emerald-700 text-white">
              <Users className="size-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Explore the assessed cohort</h2>
              <p className="text-sm text-slate-500">Instant — no inference required</p>
            </div>
          </CardHeader>
          <CardContent>
            <p className="mb-4 text-sm leading-6 text-slate-600">
              Every recording in the study cohort has already been preprocessed and scored.
              Open one to see the complete report: risk scores, signal quality with per-ICA
              rejection reasons, band power, explainability and the z_eeg descriptor.
            </p>

            <div className="space-y-2">
              {featured.map((subject) => (
                <button
                  key={subject.subject_id}
                  type="button"
                  disabled={busy}
                  onClick={() => openCohortSubject(subject.subject_id)}
                  className="flex w-full items-center justify-between gap-3 rounded-lg border border-slate-200 p-3 text-left transition-colors hover:border-blue-300 hover:bg-blue-50 disabled:opacity-50"
                >
                  <span className="min-w-0">
                    <span className="block truncate font-mono text-xs font-semibold">
                      {subject.subject_id}
                    </span>
                    <span className="text-xs text-slate-500">
                      {subject.true_class} · site {subject.site} · {subject.signal_quality}
                    </span>
                  </span>
                  <span className="shrink-0 text-right">
                    <span className="block text-xs text-slate-500">top risk</span>
                    <span className="text-sm font-bold">
                      {subject.highest_risk_condition} {percent(subject.highest_risk_score)}
                    </span>
                  </span>
                </button>
              ))}
              {!featured.length && !loadError && (
                <p className="py-6 text-center text-sm text-slate-500">Loading cohort…</p>
              )}
            </div>

            <Button className="mt-5 w-full" variant="secondary" asChild>
              <Link href="/eeg/cohort">
                <Users className="size-4" />
                Browse all subjects
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center gap-3">
            <div className="grid size-11 place-items-center rounded-lg bg-blue-700 text-white">
              <BrainCircuit className="size-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Upload a recording</h2>
              <p className="text-sm text-slate-500">Full pipeline · 30–90 seconds</p>
            </div>
          </CardHeader>
          <CardContent>
            {inferenceOff ? (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900">
                <p className="font-semibold">Inference is not enabled on this deployment.</p>
                <p className="mt-1">
                  PyTorch or the TorchScript graph is missing, so uploads are disabled.
                  Cohort browsing is unaffected.
                </p>
              </div>
            ) : (
              <>
                <p className="mb-4 text-sm leading-6 text-slate-600">
                  Provide an EEGLAB <code>.set</code> file — and its <code>.fdt</code>{" "}
                  companion when the signal is stored separately. The recording is filtered,
                  ICA-cleaned, epoched and scored, then discarded.
                </p>

                {files.length === 0 ? (
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    className="flex w-full flex-col items-center gap-2 rounded-lg border-2 border-dashed border-slate-300 p-8 text-center transition-colors hover:border-blue-400 hover:bg-blue-50"
                  >
                    <UploadCloud className="size-8 text-blue-700" />
                    <span className="text-sm font-semibold">Choose .set (and .fdt)</span>
                    <span className="text-xs text-slate-500">
                      BioSemi 128 montage · up to 120 MB
                    </span>
                  </button>
                ) : (
                  <div className="space-y-2">
                    {files.map((file) => (
                      <div
                        key={file.name}
                        className="flex items-center gap-3 rounded-lg border border-blue-200 bg-blue-50 p-3"
                      >
                        <FileUp className="size-5 shrink-0 text-blue-700" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-semibold">{file.name}</div>
                          <div className="text-xs text-slate-500">
                            {(file.size / 1024 / 1024).toFixed(1)} MB
                          </div>
                        </div>
                      </div>
                    ))}
                    <Button variant="ghost" size="sm" onClick={() => setFiles([])} disabled={busy}>
                      <X className="size-4" />
                      Clear
                    </Button>
                  </div>
                )}

                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  accept=".set,.fdt"
                  className="hidden"
                  onChange={(event) => acceptFiles(event.target.files)}
                />

                {job && busy && (
                  <div className="mt-5">
                    <div className="mb-2 flex justify-between text-xs font-semibold">
                      <span>{job.stage_label || job.status}</span>
                      <span className="tabular-nums">{job.progress}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full rounded-full bg-blue-700 transition-all duration-500"
                        style={{ width: `${Math.max(job.progress, 3)}%` }}
                      />
                    </div>
                    <p className="mt-2 text-xs text-slate-500">
                      FastICA on 128 channels is the slow step. This usually takes 30–90 seconds.
                    </p>
                  </div>
                )}

                <Button
                  className="mt-5 w-full"
                  size="lg"
                  disabled={!files.length || busy}
                  onClick={analyze}
                >
                  {busy ? (
                    <>
                      <Loader2 className="size-5 animate-spin" />
                      Assessing recording
                    </>
                  ) : (
                    "Run EEG assessment"
                  )}
                </Button>
              </>
            )}

            {error && (
              <div role="alert" className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm leading-6 text-red-800">
                {error}
              </div>
            )}

            {rejection?.worst_channels && rejection.worst_channels.length > 0 && (
              <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-4">
                <h3 className="text-sm font-bold text-amber-900">
                  Noisiest electrodes in this recording
                </h3>
                <p className="mt-1 text-xs leading-5 text-amber-900">
                  An epoch is discarded when its worst single channel exceeds{" "}
                  {rejection.threshold_uv} µV, so a few bad electrodes can fail an
                  otherwise usable recording.
                  {typeof rejection.channels_over_threshold === "number" &&
                    ` ${rejection.channels_over_threshold} of ${rejection.total_channels} channels are over the limit.`}
                </p>
                <ul className="mt-3 space-y-1.5">
                  {rejection.worst_channels.slice(0, 5).map((channel) => (
                    <li key={channel.channel} className="flex items-center gap-3 text-xs">
                      <span className="w-10 font-mono font-semibold">{channel.channel}</span>
                      <span className="h-2 flex-1 overflow-hidden rounded-full bg-amber-200">
                        <span
                          className="block h-full rounded-full bg-amber-600"
                          style={{
                            width: `${Math.min(
                              (channel.peak_to_peak_uv /
                                (rejection.worst_channels?.[0]?.peak_to_peak_uv || 1)) * 100,
                              100,
                            )}%`,
                          }}
                        />
                      </span>
                      <span className="w-24 text-right tabular-nums">
                        {Math.round(channel.peak_to_peak_uv).toLocaleString()} µV
                      </span>
                    </li>
                  ))}
                </ul>
                {typeof rejection.median_epoch_peak_to_peak_uv === "number" && (
                  <p className="mt-3 text-xs text-amber-900">
                    Typical epoch amplitude{" "}
                    <strong>
                      {Math.round(rejection.median_epoch_peak_to_peak_uv).toLocaleString()} µV
                    </strong>
                    ; {rejection.epochs_surviving} of {rejection.epochs_generated} epochs
                    passed, {rejection.epochs_required} needed.
                  </p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {card && (
        <Card className="mt-6">
          <CardContent className="flex flex-wrap items-center justify-between gap-4 p-5">
            <div className="text-sm">
              <span className="font-semibold">{card.architecture}</span>
              <span className="text-slate-500">
                {" "}· {card.input_representation} input · {card.embedding_dim}-D z_eeg ·{" "}
                risk heads {card.risk_conditions.join(" / ")}
              </span>
            </div>
            <Button variant="outline" size="sm" asChild>
              <Link href="/model-card">Model card &amp; confounds</Link>
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="mt-6">
        <ResearchDisclaimer />
      </div>
    </AppShell>
  );
}
