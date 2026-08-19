"use client";
import { useState } from "react";
import { CheckCircle2, Server } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
export default function SettingsPage(){const [status,setStatus]=useState("");async function check(){setStatus("Checking...");try{const base=process.env.NEXT_PUBLIC_API_BASE_URL??"http://127.0.0.1:8000";const response=await fetch(`${base.replace(/\/$/,"")}/health/`);setStatus(response.ok?"Backend is available":"Backend returned an error");}catch{setStatus("Backend is unreachable");}}return <AppShell><PageHeader title="Settings" description="Review this browser’s research-system connection and storage behavior."/><Card className="max-w-2xl"><CardContent className="p-6"><div className="flex items-start gap-4"><Server className="size-6 text-blue-700"/><div className="flex-1"><h2 className="section-title">Backend connection</h2><p className="mt-2 text-sm leading-6 text-slate-500">API: {process.env.NEXT_PUBLIC_API_BASE_URL??"http://127.0.0.1:8000"}</p>{status&&<p className="mt-3 flex items-center gap-2 text-sm"><CheckCircle2 className="size-4 text-emerald-700"/>{status}</p>}<Button className="mt-5" variant="outline" onClick={check}>Test connection</Button></div></div></CardContent></Card></AppShell>}
