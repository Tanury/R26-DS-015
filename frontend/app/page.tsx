import Link from "next/link";
import { Activity, ArrowRight, CheckCircle2, ClipboardList, Cpu, Mic, ScanSearch } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { ResearchDisclaimer } from "@/components/research-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

// Three assessment routes, ordered by how much the operator has to supply:
// a raw recording, an already-measured value sheet, then a full EEG session.
const options = [
  { href: "/voice", cta: "Start Voice Assessment", primary: true, title: "Neurological Voice Assessment", description: "Upload or record a standardized speech sample. Gemini structures research acoustic estimates before the classifier evaluates the feature profile.", icon: Mic, color: "bg-blue-700", bullets: ["Voice stability markers", "Speech fluency and pause analysis", "Four-class ML probability output"] },
  { href: "/general", cta: "Start Biomedical Assessment", primary: false, title: "General Biomedical Risk Assessment", description: "Enter 14 quantitative biomarker measurements that were already produced by an acoustic analysis tool. No recording is captured or uploaded — the model scores the measurement sheet directly.", icon: ClipboardList, color: "bg-emerald-700", bullets: ["Exact measurement validation", "Standardized model scaling", "Risk score and recommendations"] },
  { href: "/eeg", cta: "Start EEG Assessment", primary: false, title: "EEG Neurological Risk Assessment", description: "Upload an EEGLAB recording for a full preprocessing and inference run, or browse the 115-subject BrainLat reference cohort that the encoder was evaluated on.", icon: Activity, color: "bg-violet-700", bullets: ["Independent AD, PD and MS risk scores", "Signal quality and band power reporting", "Confound disclosure with every score"] },
];

export default function OverviewPage() {
  return <AppShell><PageHeader title="Neurological Risk Analysis" description="Assess neurological risk from three independent evidence sources — a voice recording, measured biomedical markers, or an EEG session — using research machine-learning models." /><ResearchDisclaimer />
    <div className="mt-8 grid gap-6 md:grid-cols-2 xl:grid-cols-3">{options.map(({ href, cta, primary, title, description, icon: Icon, color, bullets }) => <Card key={href}><CardContent className="flex h-full flex-col p-6 sm:p-8"><div className={`grid size-12 place-items-center rounded-lg text-white ${color}`}><Icon className="size-6" /></div><h2 className="mt-6 text-2xl font-bold">{title}</h2><p className="mt-3 leading-7 text-slate-600">{description}</p><ul className="my-7 space-y-3">{bullets.map((bullet) => <li key={bullet} className="flex items-center gap-3 text-sm"><CheckCircle2 className="size-5 shrink-0 text-emerald-700" />{bullet}</li>)}</ul><Button className="mt-auto w-full" variant={primary ? "default" : "secondary"} asChild><Link href={href}>{cta} <ArrowRight className="size-4" /></Link></Button></CardContent></Card>)}</div>
    <Card className="mt-8"><CardContent className="p-6 sm:p-8"><h2 className="text-center text-2xl font-bold">How the analysis works</h2><div className="mt-8 grid gap-6 text-center sm:grid-cols-2 lg:grid-cols-4">{[[Mic,"1. Capture","Record a voice sample, enter measured markers, or upload an EEG recording."],[ScanSearch,"2. Extract","Structure acoustic biomarkers, or filter and epoch the EEG signal."],[Cpu,"3. Predict","Apply the trained model for the chosen evidence source."],[ClipboardList,"4. Review","Review scores, confounds, caveats, and next steps."]].map(([Icon,title,copy]) => { const I=Icon as typeof Mic; return <div key={String(title)}><div className="mx-auto grid size-12 place-items-center rounded-full bg-blue-50 text-blue-700"><I className="size-5" /></div><h3 className="mt-4 font-semibold">{String(title)}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{String(copy)}</p></div>})}</div></CardContent></Card>
  </AppShell>;
}
