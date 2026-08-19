import { percent } from "@/lib/utils";

const order = ["Healthy", "PD", "MS", "AD"];

export function ProbabilityBars({ probabilities }: { probabilities: Record<string, number> }) {
  const labels = [...order.filter((label) => label in probabilities), ...Object.keys(probabilities).filter((label) => !order.includes(label))];
  return (
    <div className="space-y-5">
      {labels.map((label) => {
        const value = probabilities[label] ?? 0;
        return (
          <div key={label}>
            <div className="mb-2 flex justify-between text-sm font-medium"><span>{label}</span><span>{percent(value)}</span></div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-200">
              <div className={label === "Healthy" ? "h-full rounded-full bg-emerald-700" : "h-full rounded-full bg-blue-700"} style={{ width: `${Math.max(value * 100, value ? 1.5 : 0)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
