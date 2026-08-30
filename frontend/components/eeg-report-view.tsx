"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { BrainCircuit, RefreshCw } from "lucide-react";
import { BandPatternComparison } from "@/components/band-pattern-comparison";
import { BackButton } from "@/components/back-button";
import { EegPredictionScores } from "@/components/eeg-prediction-scores";
import { EegRecommendations } from "@/components/eeg-recommendations";
import {
  BandPowerChart,
  EegQualityPanel,
  EmbeddingPanel,
  ScalpImportance,
} from "@/components/eeg-panels";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { fetchBandReference } from "@/lib/eeg-api";
import { buildEegUserGuidance } from "@/lib/eeg-prediction-guidance";
import type { BandReference, EegRiskReport } from "@/lib/eeg-types";

export function EegReportView({ report }: { report: EegRiskReport }) {
  const { risk_assessment: assessment, signal_quality: quality } = report;
  const guidance = buildEegUserGuidance(report);
  const [reference, setReference] = useState<BandReference | null>(null);

  // Cohort statistics, not part of the report. Fetched here so both the upload and
  // the cohort route get the comparison without either having to know about it; a
  // failure just drops the panel, since it is context rather than a finding.
  useEffect(() => {
    let cancelled = false;
    fetchBandReference()
      .then((value) => !cancelled && setReference(value))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-6">
      <BackButton
        href={report.source === "cohort" ? "/eeg/cohort" : "/eeg"}
        label={report.source === "cohort" ? "Back to EEG Cohort" : "Back to EEG Assessment"}
        className="mb-0"
      />
      <Card>
        <CardContent className="flex flex-col justify-between gap-5 p-6 sm:flex-row sm:items-center">
          <div className="flex items-start gap-4">
            <BrainCircuit className="mt-1 size-10 shrink-0 text-blue-700" />
            <div className="min-w-0">
              <div className="text-xs font-bold uppercase tracking-wide text-slate-500">
                EEG risk assessment · {report.source === "cohort" ? "cohort subject" : "uploaded recording"}
              </div>
              <h1 className="mt-1 break-all text-2xl font-bold sm:text-3xl">
                {report.subject_id}
              </h1>
              <p className="mt-1 text-sm text-slate-600">
                {report.dataset.name ?? "EEG recording"}
                {report.dataset.site ? ` · site ${report.dataset.site}` : ""}
                {report.dataset.true_class ? ` · labelled ${report.dataset.true_class}` : ""}
              </p>
            </div>
          </div>
          <div className="shrink-0 rounded-lg border border-slate-300 bg-slate-50 px-6 py-4 text-center">
            <div className="mt-1 text-3xl font-bold text-blue-700">
              {guidance.top.condition}
            </div>
            <div className="text-xs text-slate-500">
              {guidance.top.points} / 100 points
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="section-title">EEG Prediction Scores</h2>
          <p className="mt-1 text-sm leading-6 text-slate-500">
            All four condition patterns are shown, with the selected result emphasized by its tinted card.
          </p>
        </CardHeader>
        <CardContent>
          <EegPredictionScores conditions={guidance.conditions} />
          <p className="mt-5 border-t border-slate-200 pt-4 text-xs leading-6 text-slate-500">
            Display points are stable for this saved assessment. They are presentation scores rather
            than calibrated clinical probabilities and do not need to total 100.
          </p>
        </CardContent>
      </Card>

      <EegRecommendations recommendations={guidance.recommendations} />

      <div className="grid gap-6 lg:grid-cols-2">
        <EegQualityPanel quality={quality} />
        <BandPowerChart profile={report.band_power_profile} />
      </div>

      <BandPatternComparison
        profile={report.band_power_profile}
        reference={reference}
        condition={assessment.highest_risk_condition}
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <ScalpImportance
          explainability={report.explainability}
          condition={assessment.highest_risk_condition}
        />
        <EmbeddingPanel embedding={report.embedding} />
      </div>

      <div className="flex flex-wrap gap-3">
        <Button asChild>
          <Link href="/eeg">
            <RefreshCw className="size-4" />
            New EEG assessment
          </Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/eeg/cohort">Browse cohort</Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/eeg/prediction">Prediction &amp; Recommendations</Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/fusion"><BrainCircuit className="size-4" />Fusion Result</Link>
        </Button>
      </div>

      <ResearchDisclaimer />
      <p className="text-xs leading-6 text-slate-500">{report.clinical_disclaimer}</p>
    </div>
  );
}
