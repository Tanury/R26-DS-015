"use client";

import { useState, type ReactNode } from "react";
import { Loader2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { EegJob } from "@/lib/eeg-types";
import {
  runBiomedicalUpload,
  runEegUpload,
  runNeuroimagingUpload,
  runVoiceUpload,
} from "@/lib/fusion-uploads";
import { IMAGING_UPLOAD_MODES, type UploadCondition } from "@/lib/neuroimaging-api";
import type { HistoryItem } from "@/lib/types";

type OnResult = (item: HistoryItem) => void;

function FilePicker({
  label,
  accept,
  multiple,
  disabled,
  files,
  onChange,
}: {
  label: string;
  accept: string;
  multiple?: boolean;
  disabled?: boolean;
  files: File[];
  onChange: (files: File[]) => void;
}) {
  return (
    <label className="flex h-11 cursor-pointer items-center gap-2 rounded-md border border-dashed border-slate-300 bg-white px-3 text-sm">
      <Upload className="size-4 shrink-0 text-slate-500" />
      <span className="truncate text-slate-600">{files.length ? files.map((file) => file.name).join(", ") : label}</span>
      <input
        type="file"
        className="hidden"
        accept={accept}
        multiple={multiple}
        disabled={disabled}
        onChange={(event) => onChange(Array.from(event.target.files ?? []))}
      />
    </label>
  );
}

/** Shared run/error wrapper so every panel behaves the same way. */
function useRunner(onResult: OnResult) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function run(task: () => Promise<HistoryItem>) {
    setRunning(true);
    setError(null);
    try {
      onResult(await task());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed.");
    } finally {
      setRunning(false);
    }
  }
  return { running, error, run };
}

function RunButton({ disabled, running, label, children }: { disabled: boolean; running: boolean; label: string; children?: ReactNode }) {
  return (
    <Button type="submit" className="w-full" disabled={disabled || running}>
      {running ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
      {running ? (children ?? "Analyzing…") : label}
    </Button>
  );
}

function ErrorNote({ error }: { error: string | null }) {
  return error ? <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900">{error}</div> : null;
}

export function VoiceUploadPanel({ onResult }: { onResult: OnResult }) {
  const [files, setFiles] = useState<File[]>([]);
  const [age, setAge] = useState("");
  const [task, setTask] = useState("reading");
  const { running, error, run } = useRunner(onResult);
  return (
    <form className="mt-3 space-y-2" onSubmit={(event) => { event.preventDefault(); void run(() => runVoiceUpload(files[0], Number(age), task)); }}>
      <FilePicker label="Choose audio (WAV, MP3, M4A, OGG, WebM)" accept="audio/wav,audio/mpeg,audio/mp4,audio/x-m4a,audio/ogg,audio/webm" disabled={running} files={files} onChange={setFiles} />
      <div className="grid grid-cols-2 gap-2">
        <select className="h-11 rounded-md border border-slate-300 bg-white px-3 text-sm" aria-label="Recording task" value={task} disabled={running} onChange={(event) => setTask(event.target.value)}>
          <option value="reading">Reading</option>
          <option value="picture_description">Picture description</option>
          <option value="monologue">Monologue</option>
          <option value="sustained_vowel">Sustained vowel</option>
        </select>
        <Input type="number" min={18} max={120} placeholder="Patient age" aria-label="Patient age" value={age} disabled={running} onChange={(event) => setAge(event.target.value)} />
      </div>
      <RunButton disabled={!files.length || !age} running={running} label="Analyze voice" />
      <ErrorNote error={error} />
    </form>
  );
}

export function BiomedicalUploadPanel({ onResult }: { onResult: OnResult }) {
  const [files, setFiles] = useState<File[]>([]);
  const { running, error, run } = useRunner(onResult);
  return (
    <form className="mt-3 space-y-2" onSubmit={(event) => { event.preventDefault(); void run(() => runBiomedicalUpload(files[0])); }}>
      <FilePicker label="Choose record (PDF or photo)" accept="application/pdf,image/jpeg,image/png,image/webp,.pdf,.jpg,.jpeg,.png,.webp" disabled={running} files={files} onChange={setFiles} />
      <p className="text-xs leading-5 text-slate-500">Values are read from the record and scored as extracted; missing values are imputed. Review values on the Biomedical page if accuracy matters.</p>
      <RunButton disabled={!files.length} running={running} label="Extract and analyze" />
      <ErrorNote error={error} />
    </form>
  );
}

export function EegUploadPanel({ onResult }: { onResult: OnResult }) {
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<EegJob | null>(null);
  const { running, error, run } = useRunner(onResult);
  return (
    <form className="mt-3 space-y-2" onSubmit={(event) => { event.preventDefault(); void run(() => runEegUpload(files, setJob)); }}>
      <FilePicker label="Choose .set (and .fdt)" accept=".set,.fdt" multiple disabled={running} files={files} onChange={setFiles} />
      <RunButton disabled={!files.length} running={running} label="Analyze EEG">
        {job ? `${job.stage_label} · ${Math.round(job.progress)}%` : "Uploading…"}
      </RunButton>
      {running && <p className="text-xs text-slate-500">Preprocessing usually takes 30–90 seconds.</p>}
      <ErrorNote error={error} />
    </form>
  );
}

export function ImagingUploadPanel({ condition, onResult }: { condition: UploadCondition; onResult: OnResult }) {
  const modes = IMAGING_UPLOAD_MODES[condition];
  const [modeId, setModeId] = useState(modes[0].id);
  const [files, setFiles] = useState<Record<string, File>>({});
  const { running, error, run } = useRunner(onResult);
  const mode = modes.find((candidate) => candidate.id === modeId) ?? modes[0];
  const ready = mode.files.every((file) => files[file.key]);
  return (
    <form className="mt-2 space-y-2" onSubmit={(event) => { event.preventDefault(); void run(() => runNeuroimagingUpload(condition, mode.id, files)); }}>
      {modes.length > 1 && (
        <select
          className="h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm"
          aria-label={`${condition} scan type`}
          value={mode.id}
          disabled={running}
          onChange={(event) => { setModeId(event.target.value); setFiles({}); }}
        >
          {modes.map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.label}</option>)}
        </select>
      )}
      {mode.files.map((file) => (
        <FilePicker
          key={`${mode.id}-${file.key}`}
          label={`Choose ${file.label}`}
          accept={file.accept}
          disabled={running}
          files={files[file.key] ? [files[file.key]] : []}
          onChange={(chosen) => { if (chosen[0]) setFiles((current) => ({ ...current, [file.key]: chosen[0] })); }}
        />
      ))}
      <RunButton disabled={!ready} running={running} label={`Analyze ${condition} scan`} />
      <ErrorNote error={error} />
    </form>
  );
}
