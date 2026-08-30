import { ClipboardList } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export function VoiceRecommendations({ recommendations }: { recommendations: string[] }) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center gap-3">
          <ClipboardList className="size-6 text-emerald-700" />
          <h2 className="section-title">Recommendations Based on This Result</h2>
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          These next steps are selected from the emphasized voice pattern and extraction quality.
        </p>
        <ol className="mt-5 space-y-4 text-sm leading-6 text-slate-700">
          {recommendations.map((recommendation, index) => (
            <li key={recommendation} className="flex gap-3">
              <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-bold text-emerald-800">{index + 1}</span>
              <span>{recommendation}</span>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}
