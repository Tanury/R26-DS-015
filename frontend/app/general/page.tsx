"use client";

import { FormEvent, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  CheckCircle2,
  Dna,
  FlaskConical,
  Info,
  Loader2,
  UserRound,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { BackButton } from "@/components/back-button";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { submitGeneralAssessment } from "@/lib/api";
import { biomarkerDetails, biomarkerGroups } from "@/lib/general-assessment";
import { saveAssessment } from "@/lib/history";
import {
  biomedicalInputKeys,
  biomedicalNumericKeys,
  type BiomedicalFeatures,
  type BiomedicalInputKey,
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
  const [values, setValues] = useState<Record<BiomedicalInputKey, string>>(sampleValues);
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
        description="A complete sample record is prefilled across the model's exact 24-field clinical, genetic, and biomarker contract. Replace the examples with the participant's actual values; blank values are sent as null and imputed by the saved pipeline."
      />
      <form
        onSubmit={submit}
        className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_340px]"
      >
        <div className="space-y-6">
          <div className="flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
            <Info className="mt-0.5 size-5 shrink-0" />
            <div>
              <div className="font-semibold">Sample values are currently displayed</div>
              <p className="mt-1">Replace them with the participant&apos;s measured values before assessment. Clear a field when it is genuinely unavailable; do not treat these examples as clinical reference ranges.</p>
            </div>
          </div>
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
                            <span className="text-xs font-normal text-slate-400">optional</span>
                          </span>
                          {detail.options ? (
                            <select
                              className="h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-100"
                              value={values[key]}
                              onChange={(event) => setValues((current) => ({ ...current, [key]: event.target.value }))}
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
                                onChange={(event) => setValues((current) => ({ ...current, [key]: event.target.value }))}
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
              The initial values are examples only. Replace them with source-record values. Do not guess missing values, and match each clinical scale, laboratory method, specimen, and unit.
            </div>
            {error && <div role="alert" className="mt-5 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">{error}</div>}
            <Button type="submit" size="lg" className="mt-6 w-full" disabled={loading || invalidKeys.length > 0}>
              {loading ? <><Loader2 className="size-4 animate-spin" />Analyzing</> : "Run Risk Assessment"}
            </Button>
          </CardContent>
        </Card>
      </form>
    </AppShell>
  );
}
