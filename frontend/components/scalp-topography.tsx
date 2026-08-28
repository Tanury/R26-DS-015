/**
 * Head diagram shading the four scalp regions by occlusion importance.
 *
 * Four sectors, not an interpolated topomap. The pipeline occludes whole BioSemi
 * banks — A posterior, B right lateral, C left lateral, D frontal-central — so four
 * aggregates is the entire spatial resolution that was measured. Smoothing them into
 * a continuous scalp map would draw detail the occlusion run never produced.
 *
 * This is the one panel in the EEG module that is genuinely causal: each value is the
 * drop in the predicted condition's risk when that region is zeroed and the recording
 * is re-scored.
 */

const HEAD = { cx: 100, cy: 100, r: 78 };

function sector(fromDeg: number, toDeg: number): string {
  const point = (deg: number) => {
    const rad = (deg * Math.PI) / 180;
    return [
      HEAD.cx + HEAD.r * Math.cos(rad),
      HEAD.cy + HEAD.r * Math.sin(rad),
    ].map((n) => n.toFixed(2));
  };
  const [x1, y1] = point(fromDeg);
  const [x2, y2] = point(toDeg);
  return `M ${HEAD.cx} ${HEAD.cy} L ${x1} ${y1} A ${HEAD.r} ${HEAD.r} 0 0 1 ${x2} ${y2} Z`;
}

// SVG angles run clockwise from east with y pointing down, so 270 deg is the top of
// the head. Each region owns a 90 deg sector centred on its anatomical position.
const REGIONS = [
  { key: "frontal_central", label: "Frontal", bank: "D", d: sector(225, 315), lx: 100, ly: 54 },
  { key: "right_lateral", label: "Right", bank: "B", d: sector(315, 45), lx: 144, ly: 96 },
  { key: "posterior", label: "Posterior", bank: "A", d: sector(45, 135), lx: 100, ly: 140 },
  { key: "left_lateral", label: "Left", bank: "C", d: sector(135, 225), lx: 56, ly: 96 },
] as const;

/** Positive = the model leaned on this region. Negative = zeroing it raised the score. */
function fillFor(value: number, scale: number): { fill: string; dark: boolean } {
  const intensity = Math.min(Math.abs(value) / scale, 1);
  const alpha = 0.08 + intensity * 0.84;
  return {
    fill: value >= 0 ? `rgba(29, 78, 216, ${alpha})` : `rgba(100, 116, 139, ${alpha})`,
    dark: alpha > 0.55,
  };
}

export function ScalpTopography({
  importance,
  condition,
}: {
  importance: Record<string, number>;
  condition: string;
}) {
  const values = REGIONS.map((region) => importance[region.key]).filter(
    (value): value is number => typeof value === "number",
  );
  if (!values.length) return null;

  const scale = Math.max(...values.map(Math.abs), 0.01);

  return (
    <figure className="m-0">
      <svg
        viewBox="0 0 200 200"
        className="mx-auto block h-auto w-full max-w-[260px]"
        role="img"
        aria-label={`Scalp regions shaded by how much each contributed to the ${condition} risk score`}
      >
        {/* Nose marks the front of the head; ears fix left from right. */}
        <path d="M 90 27 L 100 6 L 110 27 Z" className="fill-slate-100 stroke-slate-400" strokeWidth="1.5" />
        <path d="M 24 84 C 12 90 12 110 24 116" className="fill-slate-100 stroke-slate-400" strokeWidth="1.5" />
        <path d="M 176 84 C 188 90 188 110 176 116" className="fill-slate-100 stroke-slate-400" strokeWidth="1.5" />

        {REGIONS.map((region) => {
          const value = importance[region.key];
          if (typeof value !== "number") return null;
          const { fill, dark } = fillFor(value, scale);
          return (
            <g key={region.key}>
              <path d={region.d} fill={fill} stroke="#ffffff" strokeWidth="1.5" />
              <text
                x={region.lx}
                y={region.ly}
                textAnchor="middle"
                className={`text-[9px] font-bold ${dark ? "fill-white" : "fill-slate-700"}`}
              >
                {region.label}
              </text>
              <text
                x={region.lx}
                y={region.ly + 11}
                textAnchor="middle"
                className={`text-[8px] font-semibold tabular-nums ${
                  dark ? "fill-white/90" : "fill-slate-600"
                }`}
              >
                {region.bank} · {value >= 0 ? "+" : ""}
                {value.toFixed(2)}
              </text>
            </g>
          );
        })}

        <circle
          cx={HEAD.cx}
          cy={HEAD.cy}
          r={HEAD.r}
          fill="none"
          className="stroke-slate-400"
          strokeWidth="2"
        />
      </svg>

      <figcaption className="mt-3 space-y-2">
        <div className="flex items-center justify-center gap-4 text-[11px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="size-3 rounded-sm bg-blue-700" /> relied on
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-3 rounded-sm bg-slate-400" /> score rose when removed
          </span>
        </div>
        <p className="text-center text-[11px] leading-5 text-slate-500">
          Viewed from above, nose up, subject&apos;s left on the left. Sectors are BioSemi
          banks A–D; shading is the drop in {condition} risk when that bank is zeroed and
          the recording re-scored.
        </p>
      </figcaption>
    </figure>
  );
}
