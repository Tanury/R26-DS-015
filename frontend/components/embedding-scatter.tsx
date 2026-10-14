"use client";

import { useMemo } from "react";
import type { ProjectionPoint } from "@/lib/eeg-types";

const CLASS_COLOR: Record<string, string> = {
  HC: "#0f766e",
  AD: "#dc2626",
  PD: "#d97706",
  MS: "#2563eb",
};

/**
 * PCA scatter of subject-level z_eeg.
 *
 * Plain inline SVG — the project has no charting dependency and adding one for a
 * scatter of ~140 points would not earn its weight.
 */
export function EmbeddingScatter({
  points,
  selected,
  onSelect,
  note,
  explainedVariance,
  agreement,
}: {
  points: ProjectionPoint[];
  selected?: string | null;
  onSelect?: (subjectId: string) => void;
  note?: string;
  explainedVariance?: number[] | null;
  agreement?: Record<string, number>;
}) {
  const layout = useMemo(() => {
    if (!points.length) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const pad = 34;
    const width = 720;
    const height = 420;
    return {
      width,
      height,
      project: (p: ProjectionPoint) => ({
        cx: pad + ((p.x - minX) / spanX) * (width - pad * 2),
        cy: height - pad - ((p.y - minY) / spanY) * (height - pad * 2),
      }),
    };
  }, [points]);

  if (!layout) {
    return <p className="py-10 text-center text-sm text-slate-500">No projection available.</p>;
  }

  const classes = [...new Set(points.map((p) => p.true_class))].filter(Boolean).sort();
  const counts = classes.map(
    (name) => [name, points.filter((p) => p.true_class === name).length] as const,
  );
  // PC1 + PC2 of a 256-D embedding: worth stating, because a low figure means the
  // geometry the reader is looking at is a thin slice of what the encoder encodes.
  const captured = explainedVariance?.length
    ? explainedVariance.slice(0, 2).reduce((sum, value) => sum + value, 0)
    : null;

  const overall = agreement?.overall;

  return (
    <div>
      {/* Stated above the plot, not below it. Two components of a 256-D space carry
          a fraction of the variance, so an unstructured-looking scatter is the
          expected appearance of a well-separated embedding — a reader who sees the
          picture first and the caveat second has already drawn the wrong conclusion. */}
      {overall !== undefined && (
        <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
          <div className="text-sm font-bold text-blue-900">
            {Math.round(overall * 100)}% of subjects have a same-class nearest neighbour
          </div>
          <p className="mt-1 text-xs leading-5 text-slate-600">
            Measured in the full 256-D embedding, where the encoder actually operates.
            The scatter below shows two principal components
            {captured !== null ? ` carrying ${Math.round(captured * 100)}% of the variance` : ""},
            so visible spread understates the separation.
          </p>
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-xs">
            {classes.map((name) => (
              agreement?.[name] === undefined ? null : (
                <span key={name} className="flex items-center gap-1.5">
                  <span
                    className="size-2.5 rounded-full"
                    style={{ backgroundColor: CLASS_COLOR[name] ?? "#64748b" }}
                  />
                  <span className="font-medium">{name}</span>
                  <span className="tabular-nums text-slate-600">
                    {Math.round(agreement[name] * 100)}%
                  </span>
                </span>
              )
            ))}
          </div>
        </div>
      )}

      <div className="mb-3 flex flex-wrap items-center gap-4">
        {counts.map(([name, count]) => (
          <span key={name} className="flex items-center gap-2 text-xs font-medium">
            <span
              className="size-3 rounded-full"
              style={{ backgroundColor: CLASS_COLOR[name] ?? "#64748b" }}
            />
            {name} <span className="tabular-nums text-slate-500">({count})</span>
          </span>
        ))}
        {captured !== null && (
          <span className="ml-auto text-xs text-slate-500">
            PC1 + PC2 capture {Math.round(captured * 100)}% of the variance
          </span>
        )}
      </div>

      <div className="overflow-x-auto rounded-lg bg-slate-50 p-2">
        <svg
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="h-[420px] w-full min-w-[560px]"
          role="img"
          aria-label="PCA projection of subject-level z_eeg embeddings"
        >
          <line x1="34" y1={layout.height - 34} x2={layout.width - 34} y2={layout.height - 34} stroke="#cbd5e1" />
          <line x1="34" y1="34" x2="34" y2={layout.height - 34} stroke="#cbd5e1" />
          <text x={layout.width / 2} y={layout.height - 8} textAnchor="middle" fontSize="11" fill="#64748b">
            PC 1
          </text>
          <text x="12" y={layout.height / 2} textAnchor="middle" fontSize="11" fill="#64748b"
                transform={`rotate(-90 12 ${layout.height / 2})`}>
            PC 2
          </text>

          {points.map((point) => {
            const { cx, cy } = layout.project(point);
            const active = selected === point.subject_id;
            return (
              <circle
                key={point.subject_id}
                cx={cx}
                cy={cy}
                r={active ? 9 : 5.5}
                fill={CLASS_COLOR[point.true_class] ?? "#64748b"}
                stroke={active ? "#0f172a" : "white"}
                strokeWidth={active ? 2.5 : 1}
                opacity={selected && !active ? 0.45 : 0.9}
                className={onSelect ? "cursor-pointer" : undefined}
                onClick={onSelect ? () => onSelect(point.subject_id) : undefined}
              >
                <title>{`${point.subject_id} · ${point.true_class} · site ${point.site}`}</title>
              </circle>
            );
          })}
        </svg>
      </div>

      {note && <p className="mt-3 text-xs leading-5 text-slate-500">{note}</p>}
    </div>
  );
}
