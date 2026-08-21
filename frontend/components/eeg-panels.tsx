import { AlertTriangle, Activity, Layers, Waves } from "lucide-react";
import { ScalpTopography } from "@/components/scalp-topography";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import type {
  EegEmbeddingSummary,
  EegExplainability,
  EegSignalQuality,
} from "@/lib/eeg-types";
import { percent } from "@/lib/utils";

const GRADE_STYLE: Record<string, string> = {
  Good: "bg-emerald-100 text-emerald-800",
  Moderate: "bg-amber-100 text-amber-800",
  Poor: "bg-red-100 text-red-800",
};

export function EegQualityPanel({ quality }: { quality: EegSignalQuality }) {
  const stats = [
    { label: "Epochs used", value: String(quality.epochs_used) },
    { label: "Survived rejection", value: percent(quality.clean_epoch_ratio) },
    { label: "Channels", value: String(quality.channels) },
    { label: "Sampling rate", value: `${Math.round(quality.sampling_rate_hz)} Hz` },
    { label: "ICA removed", value: `${quality.ica_components_removed} components` },
    { label: "Source", value: quality.source_kind.replace("_", "-") },
  ];

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <h2 className="section-title">Signal Quality</h2>
        <span
          className={`rounded-full px-3 py-1 text-xs font-bold ${
            GRADE_STYLE[quality.grade] ?? "bg-slate-100 text-slate-700"
          }`}
        >
          {quality.grade}
        </span>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {stats.map((stat) => (
            <div key={stat.label} className="rounded-lg bg-slate-50 p-3">
              <dt className="text-xs font-semibold text-slate-500">{stat.label}</dt>
              <dd className="mt-1 text-lg font-bold text-slate-900">{stat.value}</dd>
            </div>
          ))}
        </dl>

        {quality.warnings.length > 0 && (
          <ul className="mt-4 space-y-2">
            {quality.warnings.map((warning) => (
              <li
                key={warning}
                className="flex gap-2 rounded-md bg-amber-50 p-3 text-xs leading-5 text-amber-900"
              >
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                {warning}
              </li>
            ))}
          </ul>
        )}

        {quality.ica_rejections.length > 0 && (
          <details className="mt-4 rounded-lg border border-slate-200 p-3">
            <summary className="cursor-pointer text-sm font-semibold text-slate-700">
              Why {quality.ica_rejections.length} ICA component
              {quality.ica_rejections.length > 1 ? "s were" : " was"} removed
            </summary>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[420px] text-left text-xs">
                <thead className="text-slate-500">
                  <tr>
                    {["Component", "Criteria fired", "Kurtosis", "Frontal r", "HF ratio"].map(
                      (heading) => (
                        <th key={heading} className="py-1.5 pr-4 font-semibold">
                          {heading}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {quality.ica_rejections.map((rejection) => (
                    <tr key={rejection.component}>
                      <td className="py-1.5 pr-4 font-mono">{rejection.component}</td>
                      <td className="py-1.5 pr-4">{rejection.criteria.join(", ")}</td>
                      <td className="py-1.5 pr-4 tabular-nums">
                        {rejection.kurtosis?.toFixed(2) ?? "—"}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums">
                        {rejection.frontal_corr?.toFixed(2) ?? "—"}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums">
                        {rejection.hf_power_ratio?.toFixed(2) ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

const BAND_ORDER = ["delta", "theta", "alpha", "beta", "low_gamma"];
const BAND_HINT: Record<string, string> = {
  delta: "0.5–4 Hz",
  theta: "4–8 Hz · elevated in AD",
  alpha: "8–13 Hz · reduced in AD",
  beta: "13–30 Hz · disrupted in PD",
  low_gamma: "30–40 Hz · altered in MS",
};

export function BandPowerChart({ profile }: { profile: Record<string, number> }) {
  const bands = BAND_ORDER.filter((band) => band in profile);
  const max = Math.max(...bands.map((band) => profile[band]), 0.01);
  const ratio = profile.theta_alpha_ratio;

  return (
    <Card>
      <CardHeader>
        <h2 className="section-title">Relative Band Power</h2>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {bands.map((band) => (
            <div key={band}>
              <div className="mb-1.5 flex items-baseline justify-between text-sm">
                <span className="font-semibold capitalize">{band.replace("_", " ")}</span>
                <span className="text-xs text-slate-500">{BAND_HINT[band]}</span>
                <span className="tabular-nums font-medium">{percent(profile[band])}</span>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full rounded-full bg-blue-700"
                  style={{ width: `${(profile[band] / max) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {typeof ratio === "number" && (
          <div className="mt-5 flex items-center gap-3 rounded-lg bg-slate-50 p-4">
            <Waves className="size-5 shrink-0 text-blue-700" />
            <div>
              <div className="text-xs font-semibold text-slate-500">Theta / alpha ratio</div>
              <div className="text-2xl font-bold tabular-nums">{ratio.toFixed(2)}</div>
            </div>
            <p className="ml-auto max-w-xs text-xs leading-5 text-slate-500">
              The most-cited summary of AD-type EEG slowing. Higher values indicate more
              theta relative to alpha.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ScalpImportance({
  explainability,
  condition,
}: {
  explainability: EegExplainability;
  condition: string;
}) {
  const regions = Object.entries(explainability.scalp_region_importance);
  const bands = Object.entries(explainability.band_importance);

  // The training run exported occlusion for its demo subjects only, so most cohort
  // rows have nothing here. Saying so beats the card silently vanishing, which reads
  // as though the question was never asked.
  if (!regions.length && !bands.length) {
    return (
      <Card>
        <CardHeader>
          <h2 className="section-title">What Drove This Score</h2>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3 rounded-lg bg-slate-50 p-4">
            <AlertTriangle className="mt-0.5 size-5 shrink-0 text-slate-400" />
            <p className="text-xs leading-6 text-slate-600">
              Occlusion attribution was not computed for this subject. The training run
              exported it for one demo subject per class only; every uploaded recording
              gets it computed live. Nothing is inferred here in its absence.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const scale = Math.max(
    ...[...regions, ...bands].map(([, value]) => Math.abs(value)),
    0.01,
  );

  const row = ([name, value]: [string, number]) => (
    <div key={name} className="flex items-center gap-3">
      <span className="w-32 shrink-0 text-xs font-medium capitalize text-slate-600">
        {name.replace(/_/g, " ")}
      </span>
      <div className="relative h-2.5 flex-1 rounded-full bg-slate-200">
        <div
          className={`absolute top-0 h-full rounded-full ${
            value >= 0 ? "left-1/2 bg-blue-700" : "right-1/2 bg-slate-400"
          }`}
          style={{ width: `${(Math.abs(value) / scale) * 50}%` }}
        />
        <div className="absolute left-1/2 top-[-3px] h-[16px] w-px bg-slate-400" />
      </div>
      <span className="w-14 text-right text-xs tabular-nums text-slate-600">
        {value >= 0 ? "+" : ""}
        {value.toFixed(3)}
      </span>
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <h2 className="section-title">What Drove This Score</h2>
      </CardHeader>
      <CardContent className="space-y-5">
        {regions.length > 0 && (
          <div>
            <ScalpTopography
              importance={explainability.scalp_region_importance}
              condition={condition}
            />
            <h3 className="mb-3 mt-6 flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Layers className="size-4 text-blue-700" /> Scalp region
            </h3>
            <div className="space-y-2.5">{regions.map(row)}</div>
          </div>
        )}
        {bands.length > 0 && (
          <div>
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Activity className="size-4 text-blue-700" /> Frequency band
            </h3>
            <div className="space-y-2.5">{bands.map(row)}</div>
          </div>
        )}
        <p className="text-xs leading-5 text-slate-500">{explainability.method}. Positive
          values mean the model relied on that region or band; zeroing it lowered the score.</p>
      </CardContent>
    </Card>
  );
}

export function EmbeddingPanel({ embedding }: { embedding: EegEmbeddingSummary }) {
  const centroids = Object.entries(embedding.cosine_to_class_centroids);
  return (
    <Card>
      <CardHeader>
        <h2 className="section-title">z_eeg Embedding</h2>
      </CardHeader>
      <CardContent>
        <p className="mb-4 text-xs leading-5 text-slate-500">
          The 256-dimensional vector handed to the multimodal fusion engine alongside
          z_img and z_bio. An all-zero vector with availability flag 0 means EEG was absent.
        </p>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "Dimensions", value: String(embedding.dim) },
            { label: "L2 norm", value: embedding.l2_norm.toFixed(3) },
            { label: "Availability", value: embedding.availability_flag === 1 ? "Present" : "Absent" },
            { label: "Epoch consistency", value: embedding.consistency.toFixed(3) },
          ].map((item) => (
            <div key={item.label} className="rounded-lg bg-slate-50 p-3">
              <dt className="text-xs font-semibold text-slate-500">{item.label}</dt>
              <dd className="mt-1 text-base font-bold tabular-nums">{item.value}</dd>
            </div>
          ))}
        </dl>

        {centroids.length > 0 && (
          <div className="mt-5">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">
              Cosine similarity to class centroids
            </h3>
            <div className="space-y-2.5">
              {centroids.map(([name, value]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="w-10 shrink-0 text-xs font-semibold">{name}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={
                        name === embedding.nearest_centroid
                          ? "h-full rounded-full bg-blue-700"
                          : "h-full rounded-full bg-slate-400"
                      }
                      style={{ width: `${Math.max(value * 100, 1)}%` }}
                    />
                  </div>
                  <span className="w-12 text-right text-xs tabular-nums">{value.toFixed(3)}</span>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Similarity is bounded in [0, 1] because the projection head applies ReLU before
              L2 normalization, confining embeddings to the non-negative orthant.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
