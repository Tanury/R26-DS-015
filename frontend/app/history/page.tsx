"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Download, Search } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { setCurrentResult, useStoredHistory } from "@/lib/history";
import type { HistoryItem } from "@/lib/types";
import { percent } from "@/lib/utils";

export default function HistoryPage(){
  const router=useRouter(); const items=useStoredHistory(); const [search,setSearch]=useState(""); const [type,setType]=useState("All");
  const filtered=useMemo(()=>items.filter(item=>(type==="All"||item.type===type)&&item.id.toLowerCase().includes(search.toLowerCase())),[items,search,type]);
  function view(item:HistoryItem){setCurrentResult(item);router.push(item.type==="Voice"?"/voice/results":"/general/results");}
  function download(item:HistoryItem){const blob=new Blob([JSON.stringify(item,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const anchor=document.createElement("a");anchor.href=url;anchor.download=`${item.id}.json`;anchor.click();URL.revokeObjectURL(url);}
  return <AppShell><PageHeader title="Assessment History" description="Review locally saved neurological speech screening results from this browser." /><Card><CardContent className="p-4 sm:p-5"><div className="flex flex-col gap-3 sm:flex-row"><div className="relative flex-1"><Search className="absolute left-3 top-3 size-5 text-slate-400"/><Input className="pl-10" placeholder="Search by assessment ID" value={search} onChange={e=>setSearch(e.target.value)} /></div><select className="h-11 rounded-md border border-slate-300 bg-white px-4 text-sm" value={type} onChange={e=>setType(e.target.value)}><option>All</option><option>Voice</option><option>General</option></select></div></CardContent></Card>
    <Card className="mt-6 overflow-hidden"><div className="overflow-x-auto"><table className="w-full min-w-[800px] text-left text-sm"><thead className="bg-slate-100 text-xs uppercase text-slate-500"><tr>{["Date","Assessment ID","Type","Result","Risk Score","Confidence","Actions"].map(label=><th key={label} className="px-5 py-4 font-semibold">{label}</th>)}</tr></thead><tbody className="divide-y divide-slate-200">{filtered.map(item=><tr key={item.id} className="hover:bg-slate-50"><td className="px-5 py-4">{new Date(item.createdAt).toLocaleDateString()}</td><td className="px-5 py-4 font-mono text-xs">{item.id}</td><td className="px-5 py-4">{item.type}</td><td className="px-5 py-4"><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${item.prediction.predicted_class==="Healthy"?"bg-emerald-100 text-emerald-800":"bg-red-100 text-red-800"}`}>{item.prediction.predicted_class}</span></td><td className="px-5 py-4">{percent(item.prediction.risk_score)}</td><td className="px-5 py-4">{percent(item.prediction.confidence_score)}</td><td className="px-5 py-4"><div className="flex items-center gap-1"><Button variant="ghost" size="sm" onClick={()=>view(item)}>View</Button><Button variant="ghost" size="icon" onClick={()=>download(item)} aria-label={`Download ${item.id}`}><Download className="size-4" /></Button></div></td></tr>)}</tbody></table>{filtered.length===0&&<div className="p-12 text-center text-sm text-slate-500">No assessments match these filters. Complete an assessment to populate history.</div>}</div></Card>
  </AppShell>;
}
