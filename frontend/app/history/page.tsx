"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Download, Eye, Search } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { setCurrentResult, useStoredHistory } from "@/lib/history";
import type { HistoryItem } from "@/lib/types";
import { percent } from "@/lib/utils";
import { buildVoiceResultGuidance } from "@/lib/voice-result-guidance";

export default function HistoryPage(){
  const router=useRouter(); const items=useStoredHistory(); const [search,setSearch]=useState(""); const [type,setType]=useState("All");
  const filtered=useMemo(()=>items.filter(item=>(type==="All"||item.type===type)&&item.id.toLowerCase().includes(search.toLowerCase())),[items,search,type]);
  const routeFor={Voice:"/voice/results",General:"/general/results",EEG:"/eeg/results"} as const;
  // The stored type stays "General" so history saved by earlier builds still
  // resolves; only the label shown to the operator changed.
  const typeLabel={Voice:"Voice",General:"Biomedical",EEG:"EEG"} as const;
  function view(item:HistoryItem){setCurrentResult(item);router.push(routeFor[item.type]);}
  // EEG rows carry three independent risk scores instead of one class plus a
  // softmax, so each type renders its own result / score / confidence cells.
  function resultCell(item:HistoryItem){
    if(item.type==="EEG"&&item.eegReport){const a=item.eegReport.risk_assessment;const top=a.conditions[a.highest_risk_condition];
      return {label:`${a.highest_risk_condition} risk`,tone:top&&top.risk_band!=="Low"?"bg-red-100 text-red-800":"bg-emerald-100 text-emerald-800",score:top?percent(top.risk_score):"-",confidence:top?top.risk_band:"-"};}
    if(item.type==="Voice"&&item.prediction){const guidance=buildVoiceResultGuidance(item.id,item.prediction,item.voiceDetails);const top=guidance.top;
      return {label:top.condition,tone:top.condition==="Healthy"?"bg-emerald-100 text-emerald-800":"bg-red-100 text-red-800",score:`${top.points} / 100`,confidence:"Primary"};}
    if(item.prediction){const p=item.prediction;
      return {label:p.predicted_class,tone:p.predicted_class==="Healthy"?"bg-emerald-100 text-emerald-800":"bg-red-100 text-red-800",score:percent(p.risk_score),confidence:percent(p.confidence_score)};}
    return {label:"-",tone:"bg-slate-100 text-slate-700",score:"-",confidence:"-"};
  }
  function download(item:HistoryItem){const blob=new Blob([JSON.stringify(item,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const anchor=document.createElement("a");anchor.href=url;anchor.download=`${item.id}.json`;anchor.click();URL.revokeObjectURL(url);}
  return <AppShell><PageHeader title="Assessment History" description="Review locally saved voice, biomedical, and EEG screening results from this browser." /><Card><CardContent className="p-4 sm:p-5"><div className="flex flex-col gap-3 sm:flex-row"><div className="relative flex-1"><Search className="absolute left-3 top-3 size-5 text-slate-400"/><Input className="pl-10" placeholder="Search by assessment ID" value={search} onChange={e=>setSearch(e.target.value)} /></div><select className="h-11 rounded-md border border-slate-300 bg-white px-4 text-sm" value={type} onChange={e=>setType(e.target.value)}><option>All</option><option>Voice</option><option value="General">Biomedical</option><option>EEG</option></select></div></CardContent></Card>
    <Card className="mt-6 overflow-hidden"><div className="overflow-x-auto"><table className="w-full min-w-[860px] text-left text-sm"><thead className="bg-slate-100 text-xs uppercase text-slate-500"><tr>{["Date","Assessment ID","Type","Result","Score","Level","Actions"].map(label=><th key={label} className="px-5 py-4 font-semibold">{label}</th>)}</tr></thead><tbody className="divide-y divide-slate-200">{filtered.map(item=>{const cell=resultCell(item);const canView=item.type==="EEG"?Boolean(item.eegReport):Boolean(item.prediction);return <tr key={item.id} className="hover:bg-slate-50"><td className="px-5 py-4">{new Date(item.createdAt).toLocaleDateString()}</td><td className="px-5 py-4 font-mono text-xs">{item.id}</td><td className="px-5 py-4">{typeLabel[item.type]}</td><td className="px-5 py-4"><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${cell.tone}`}>{cell.label}</span></td><td className="px-5 py-4">{cell.score}</td><td className="px-5 py-4">{cell.confidence}</td><td className="px-5 py-4"><div className="flex items-center gap-2 whitespace-nowrap"><Button type="button" size="sm" disabled={!canView} onClick={()=>view(item)} aria-label={`View result ${item.id}`}><Eye className="size-4" />View Result</Button><Button type="button" variant="outline" size="icon" onClick={()=>download(item)} aria-label={`Download ${item.id}`} title="Download result JSON"><Download className="size-4" /></Button></div></td></tr>;})}</tbody></table>{filtered.length===0&&<div className="p-12 text-center text-sm text-slate-500">No assessments match these filters. Complete an assessment to populate history.</div>}</div></Card>
  </AppShell>;
}
