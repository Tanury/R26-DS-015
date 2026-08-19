import Link from "next/link";
import { ArrowRight, CheckCircle2, ClipboardList, Cpu, Mic, ScanSearch } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const options = [
  { href: "/voice", title: "Neurological Voice Assessment", description: "Upload or record a standardized speech sample. Gemini structures research acoustic estimates before the classifier evaluates the feature profile.", icon: Mic, color: "bg-blue-700", bullets: ["Voice stability markers", "Speech fluency and pause analysis", "Four-class ML probability output"] },
  { href: "/general", title: "General Speech Risk Assessment", description: "Enter the 14 supported speech features directly for a controlled model assessment and complete probability distribution.", icon: ClipboardList, color: "bg-emerald-700", bullets: ["Exact feature validation", "Standardized model scaling", "Risk score and recommendations"] },
];

export default function OverviewPage() {
  return <AppShell><PageHeader title="Neurological Speech Analysis" description="Analyze speech biomarkers and evaluate neurological risk patterns using research machine-learning models." /><ResearchDisclaimer />
    <div className="mt-8 grid gap-6 lg:grid-cols-2">{options.map(({ href, title, description, icon: Icon, color, bullets }) => <Card key={href}><CardContent className="flex h-full flex-col p-6 sm:p-8"><div className={`grid size-12 place-items-center rounded-lg text-white ${color}`}><Icon className="size-6" /></div><h2 className="mt-6 text-2xl font-bold sm:text-3xl">{title}</h2><p className="mt-3 leading-7 text-slate-600">{description}</p><ul className="my-7 space-y-3">{bullets.map((bullet) => <li key={bullet} className="flex items-center gap-3 text-sm"><CheckCircle2 className="size-5 text-emerald-700" />{bullet}</li>)}</ul><Button className="mt-auto w-full" variant={href === "/voice" ? "default" : "secondary"} asChild><Link href={href}>Start {href === "/voice" ? "Voice" : "General"} Assessment <ArrowRight className="size-4" /></Link></Button></CardContent></Card>)}</div>
    <Card className="mt-8"><CardContent className="p-6 sm:p-8"><h2 className="text-center text-2xl font-bold">How the analysis works</h2><div className="mt-8 grid gap-6 text-center sm:grid-cols-2 lg:grid-cols-4">{[[Mic,"1. Capture","Provide a standardized voice sample or features."],[ScanSearch,"2. Extract","Structure the supported speech biomarkers."],[Cpu,"3. Predict","Apply scaling and model classification."],[ClipboardList,"4. Review","Review probabilities, caveats, and next steps."]].map(([Icon,title,copy]) => { const I=Icon as typeof Mic; return <div key={String(title)}><div className="mx-auto grid size-12 place-items-center rounded-full bg-blue-50 text-blue-700"><I className="size-5" /></div><h3 className="mt-4 font-semibold">{String(title)}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{String(copy)}</p></div>})}</div></CardContent></Card>
  </AppShell>;
}
