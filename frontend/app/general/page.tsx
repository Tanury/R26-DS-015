"use client";

import { FormEvent, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  CheckCircle2,
  Dna,
  FileText,
  FlaskConical,
  Info,
  Loader2,
  Sparkles,
  Upload,
  UserRound,
  X,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { extractBiomedicalRecord, submitGeneralAssessment } from "@/lib/api";
import { biomarkerDetails, biomarkerGroups } from "@/lib/general-assessment";
import { saveAssessment } from "@/lib/history";
import {
  biomedicalInputKeys,
  biomedicalNumericKeys,
  type BiomedicalFeatures,
  type BiomedicalInputKey,
  type BiomedicalRecordExtraction,
} from "@/lib/types";

const sampleValues: Record<BiomedicalInputKey, string> = {
  age: "65",
  sex: "Female",
  education_years: "14",
  bmi: "25",
  family_history_pd: "0",
  systolic_bp: "125",
  diastolic_bp: "80",
  cognitive_screen_score_0_30: "27",
  rem_sleep_score: "4",
  updrs_part_i: "4",
  updrs_part_ii: "5",
  updrs_part_iii: "12",
  updrs_part_iv: "0",
  schwab_england_adl: "90",
  apoe_e4_count: "1",
  gba_variant_carrier: "0",
  amyloid_beta_42_40_ratio: "0.08",
  t_tau_pg_ml: "300",
  p_tau181_pg_ml: "50",
  nfl_pg_ml: "800",
  gfap_pg_ml: "200",
  alpha_synuclein_pg_ml: "1200",
  gdf15_pg_ml: "800",
  crp40_copy_number: "2000",
};

const numericKeys = new Set<string>(biomedicalNumericKeys);
const groupIcons = [UserRound, Activity, Dna, FlaskConical];
const acceptedRecordTypes = new Set([
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/webp",
]);
const maxRecordBytes = 15 * 1024 * 1024;
const emptyValues = Object.fromEntries(
  biomedicalInputKeys.map((key) => [key, ""]),
) as Record<BiomedicalInputKey, string>;

function isValidValue(key: BiomedicalInputKey, value: string) {
  if (!value.trim()) return true;
  if (!numericKeys.has(key)) return true;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return false;
  const detail = biomarkerDetails[key];
  if (detail.min !== undefined && numeric < detail.min) return false;
  if (detail.max !== undefined && numeric > detail.max) return false;
  return true;
}

export default function GeneralAssessmentPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [values, setValues] = useState<Record<BiomedicalInputKey, string>>(sampleValues);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [extraction, setExtraction] = useState<BiomedicalRecordExtraction | null>(null);
  const [importedKeys, setImportedKeys] = useState<Set<BiomedicalInputKey>>(new Set());
  const [extracting, setExtracting] = useState(false);
  const [extractionError, setExtractionError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const providedCount = useMemo(
    () => biomedicalInputKeys.filter((key) => values[key].trim()).length,
    [values],
  );
  const invalidKeys = useMemo(
    () => biomedicalInputKeys.filter((key) => !isValidValue(key, values[key])),
    [values],
  );

  async function importRecord(file: File) {
    setExtractionError("");
    setError("");

    if (!acceptedRecordTypes.has(file.type)) {
      setUploadedFile(null);
      setExtractionError("Upload a PDF, JPG, PNG, or WebP file.");
      return;
    }
    if (file.size === 0) {
      setUploadedFile(null);
      setExtractionError("The selected file is empty.");
      return;
    }
    if (file.size > maxRecordBytes) {
      setUploadedFile(null);
      setExtractionError("The selected file must be 15 MB or smaller.");
      return;
    }

    setUploadedFile(file);
    setExtracting(true);
    try {
      const result = await extractBiomedicalRecord(file);
      setValues(Object.fromEntries(
        biomedicalInputKeys.map((key) => {
          const value = result.features[key];
          return [key, value === null ? "" : String(value)];
        }),
      ) as Record<BiomedicalInputKey, string>);
      setImportedKeys(new Set(
        biomedicalInputKeys.filter((key) => result.features[key] !== null),
      ));
      setExtraction(result);
    } catch (reason) {
      setUploadedFile(null);
      setExtractionError(
        reason instanceof Error ? reason.message : "The record could not be analyzed.",
      );
    } finally {
      setExtracting(false);
    }
  }

  function updateValue(key: BiomedicalInputKey, value: string) {
    setValues((current) => ({ ...current, [key]: value }));
    setImportedKeys((current) => {
      if (!current.has(key)) return current;
      const next = new Set(current);
      next.delete(key);
      return next;
    });
  }

  function clearImportedRecord() {
    setUploadedFile(null);
    setExtraction(null);
    setExtractionError("");
    setImportedKeys(new Set());
    setValues(emptyValues);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (invalidKeys.length) {
      setError("Correct the highlighted values before running the assessment.");
      return;
    }

    const biomarkers = Object.fromEntries(
      biomedicalInputKeys.map((key) => {
        const value = values[key].trim();
        if (!value) return [key, null];
        return [key, key === "sex" ? value : Number(value)];
      }),
    ) as BiomedicalFeatures;

    setLoading(true);
    try {
      const prediction = await submitGeneralAssessment(biomarkers);
      saveAssessment("General", prediction, { biomarkers });
      router.push("/general/results");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Assessment failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <BackButton href="/" label="Back to Assessments" />
      <PageHeader
        title="General Biomedical Risk Assessment"
        description="Upload a biomedical PDF or photo to fill the 24-field form with Gemini, or enter values manually. Always verify imported values against the source; blank fields are sent as null and imputed by the saved pipeline."
      />
      <form
        onSubmit={submit}
        className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_340px]"
      >
        <div className="space-y-6">
          <Card className="overflow-hidden border-blue-200">
            <CardContent className="p-0">
              <div className="border-b border-blue-100 bg-blue-50 px-6 py-5">
                <div className="flex items-start gap-3">
                  <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-blue-700 text-white">
                    <Sparkles className="size-5" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-slate-950">Fill from a document or photo</h2>
                    <p className="mt-1 text-sm leading-6 text-slate-600">
                      Gemini reads supported values and places them into the form for your review. PDF, JPG, PNG, and WebP files up to 15 MB are accepted.
                    </p>
                  </div>
                </div>
              </div>
              <div className="p-6">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf,image/jpeg,image/png,image/webp,.pdf,.jpg,.jpeg,.png,.webp"
                  className="sr-only"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void importRecord(file);
                    event.target.value = "";
                  }}
                  aria-label="Upload biomedical document or photo"
                />
                <div
                  className={`rounded-xl border-2 border-dashed px-5 py-8 text-center transition-colors ${extracting ? "border-blue-300 bg-blue-50" : "border-slate-300 hover:border-blue-400 hover:bg-slate-50"}`}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    const file = event.dataTransfer.files[0];
                    if (file && !extracting) void importRecord(file);
                  }}
                >
                  {extracting ? (
                    <>
                      <Loader2 className="mx-auto size-8 animate-spin text-blue-700" />
                      <p className="mt-3 font-semibold text-slate-900">Reading {uploadedFile?.name}</p>
                      <p className="mt-1 text-sm text-slate-500">Extracting and validating the 24 supported fields…</p>
                    </>
                  ) : extraction ? (
                    <>
                      <CheckCircle2 className="mx-auto size-8 text-emerald-700" />
                      <p className="mt-3 font-semibold text-slate-900">{extraction.filename}</p>
                      <p className="mt-1 text-sm text-slate-600">
                        {extraction.extracted_count} of 24 fields imported · {extraction.extraction_quality} extraction
                      </p>
                      <div className="mt-4 flex flex-wrap justify-center gap-2">
                        <Button type="button" variant="outline" onClick={() => fileInputRef.current?.click()}>
                          <Upload className="size-4" />Replace file
                        </Button>
                        <Button type="button" variant="ghost" onClick={clearImportedRecord}>
                          <X className="size-4" />Clear imported values
                        </Button>
                      </div>
                    </>
                  ) : (
                    <>
                      <FileText className="mx-auto size-8 text-blue-700" />
                      <p className="mt-3 font-semibold text-slate-900">Drop a record here, or choose a file</p>
                      <p className="mt-1 text-sm text-slate-500">Extraction begins automatically after selection.</p>
                      <Button type="button" className="mt-4" onClick={() => fileInputRef.current?.click()}>
                        <Upload className="size-4" />Choose document or photo
                      </Button>
                    </>
                  )}
                </div>
                {extraction?.quality_notes.length ? (
                  <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
                    <div className="font-semibold">Review notes</div>
                    <ul className="mt-2 list-disc space-y-1 pl-5">
                      {extraction.quality_notes.map((note) => <li key={note}>{note}</li>)}
                    </ul>
                  </div>
                ) : null}
                {extraction && (
                  <p className="mt-4 text-xs leading-5 text-slate-500">{extraction.extraction_disclaimer}</p>
                )}
                {extractionError && (
                  <div role="alert" className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">{extractionError}</div>
                )}
                <p className="mt-4 text-xs leading-5 text-slate-500">
                  The file is sent to the configured Gemini API through the server. Upload only records you are permitted to process with that service.
                </p>
              </div>
            </CardContent>
          </Card>
          {!extraction && !extracting && (
            <div className="flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
              <Info className="mt-0.5 size-5 shrink-0" />
              <div>
                <div className="font-semibold">Sample values are currently displayed</div>
                <p className="mt-1">Replace them manually or upload a source record. An upload replaces all samples, leaving unreported fields blank so no example value is mistaken for patient data.</p>
              </div>
            </div>
          )}
          {biomarkerGroups.map((group, groupIndex) => {
            const Icon = groupIcons[groupIndex];
            return (
              <Card key={group.title}>
                <CardContent className="p-6">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <h2 className="section-title flex items-center gap-2">
                      <Icon className="size-6 text-blue-700" />
                      {group.title}
                    </h2>
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                      {group.keys.length} fields
                    </span>
                  </div>
                  <p className="mb-6 text-sm leading-6 text-slate-500">{group.description}</p>
                  <div className="grid gap-5 md:grid-cols-2">
                    {group.keys.map((key) => {
                      const detail = biomarkerDetails[key];
                      const valid = isValidValue(key, values[key]);
                      return (
                        <label key={key} className="block">
                          <span className="mb-2 flex items-center gap-2 text-sm font-semibold">
                            {detail.label}
                            <Info className="size-3.5 text-slate-400" aria-label={detail.description} />
                            {importedKeys.has(key) && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-emerald-800">Imported</span>}
                            <span className="text-xs font-normal text-slate-400">optional</span>
                          </span>
                          {detail.options ? (
                            <select
                              className="h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
                              value={values[key]}
                              onChange={(event) => updateValue(key, event.target.value)}
                              aria-describedby={`${key}-description`}
                            >
                              <option value="">Not available</option>
                              {detail.options.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                              ))}
                            </select>
                          ) : (
                            <div className="relative">
                              <Input
                                type="number"
                                min={detail.min}
                                max={detail.max}
                                step={detail.step ?? "any"}
                                value={values[key]}
                                onChange={(event) => updateValue(key, event.target.value)}
                                aria-invalid={!valid}
                                aria-describedby={`${key}-description`}
                                className={`pr-24 ${valid ? "" : "border-red-500 focus-visible:ring-red-200"}`}
                                placeholder="Not available"
                              />
                              <span className="pointer-events-none absolute right-3 top-3 text-xs text-slate-400">{detail.unit}</span>
                            </div>
                          )}
                          <span id={`${key}-description`} className={`mt-1.5 block text-xs leading-5 ${valid ? "text-slate-500" : "text-red-700"}`}>
                            {valid ? detail.description : `Enter a value between ${detail.min ?? 0} and ${detail.max ?? "the assay's valid maximum"}.`}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <Card className="xl:sticky xl:top-24">
          <CardContent className="p-6">
            <h2 className="section-title">Input Validation</h2>
            <div className="mt-5 flex gap-3 rounded-lg bg-slate-50 p-4">
              <CheckCircle2 className={invalidKeys.length ? "size-6 shrink-0 text-red-700" : "size-6 shrink-0 text-emerald-700"} />
              <div>
                <div className="font-semibold">{providedCount} / {biomedicalInputKeys.length} values provided</div>
                <p className="mt-1 text-sm leading-6 text-slate-500">
                  {invalidKeys.length
                    ? `${invalidKeys.length} provided value${invalidKeys.length === 1 ? " is" : "s are"} outside the accepted range.`
                    : `${biomedicalInputKeys.length - providedCount} blank value${biomedicalInputKeys.length - providedCount === 1 ? " will" : "s will"} use fitted training-set imputation.`}
                </p>
              </div>
            </div>
            <div className="mt-6 h-2 overflow-hidden rounded-full bg-slate-200">
              <div className="h-full bg-emerald-700 transition-all" style={{ width: `${(providedCount / biomedicalInputKeys.length) * 100}%` }} />
            </div>
            <div className="mt-5 rounded-lg border border-blue-100 bg-blue-50 p-4 text-xs leading-5 text-slate-600">
              {extraction
                ? "Verify every imported value against the source record. Do not guess missing values, and match each clinical scale, laboratory method, specimen, and unit."
                : "The initial values are examples only. Replace them with source-record values. Do not guess missing values, and match each clinical scale, laboratory method, specimen, and unit."}
            </div>
            {error && <div role="alert" className="mt-5 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">{error}</div>}
            <Button type="submit" size="lg" className="mt-6 w-full" disabled={loading || extracting || invalidKeys.length > 0}>
              {loading ? <><Loader2 className="size-4 animate-spin" />Analyzing</> : "Run Risk Assessment"}
            </Button>
          </CardContent>
        </Card>
      </form>
    </AppShell>
  );
}
