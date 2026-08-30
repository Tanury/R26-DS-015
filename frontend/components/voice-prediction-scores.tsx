import type { VoiceClassScore } from "@/lib/voice-result-guidance";

export function VoicePredictionScores({ conditions }: { conditions: VoiceClassScore[] }) {
  return (
    <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-4">
      {conditions.map((condition) => {
        const healthy = condition.condition === "Healthy";
        return (
          <article
            key={condition.condition}
            className={`rounded-xl border p-5 ${condition.isPrimary ? healthy ? "border-emerald-300 bg-emerald-50/60" : "border-blue-300 bg-blue-50/60" : "border-slate-200 bg-white"}`}
          >
            <div>
              <div className="text-2xl font-bold">{condition.condition}</div>
              <div className="mt-1 text-xs leading-5 text-slate-500">{condition.name}</div>
            </div>
            <div className="mt-5 flex items-end justify-between">
              <span className="text-sm font-semibold text-slate-500">Result points</span>
              <span className="text-3xl font-bold tabular-nums">{condition.points}</span>
            </div>
            <div
              className="mt-2 h-3 overflow-hidden rounded-full bg-slate-200"
              role="meter"
              aria-valuenow={condition.points}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${condition.name} result points`}
            >
              <div
                className={`h-full rounded-full ${condition.isPrimary ? healthy ? "bg-emerald-700" : "bg-blue-700" : "bg-slate-500"}`}
                style={{ width: `${condition.points}%` }}
              />
            </div>
            <p className="mt-4 text-sm leading-6 text-slate-600">{condition.explanation}</p>
          </article>
        );
      })}
    </div>
  );
}
