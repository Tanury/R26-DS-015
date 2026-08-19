"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { FileAudio, Loader2, Mic, Square, UploadCloud, X } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { submitVoiceAssessment } from "@/lib/api";
import { saveAssessment } from "@/lib/history";

const steps = ["Recording", "Analysis", "Prediction", "Results"];
const passage = "The grandfather clock ticked loudly in the hallway, its brass pendulum swinging in a steady rhythm. Outside, the rain drummed softly against the glass.";

export default function VoiceAssessmentPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [age, setAge] = useState("58");
  const [task, setTask] = useState("reading");
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");

  function acceptFile(candidate: File) {
    setError("");
    if (candidate.size > 18 * 1024 * 1024) { setError("Audio files must be 18 MB or smaller."); return; }
    if (!candidate.type.startsWith("audio/")) { setError("Choose a WAV, MP3, M4A, OGG, or WebM audio file."); return; }
    setFile(candidate);
  }

  async function toggleRecording() {
    setError("");
    if (recording) { recorderRef.current?.stop(); return; }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") { setError("Microphone recording is not supported in this browser. Upload an audio file instead."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : undefined });
      chunksRef.current = [];
      recorder.ondataavailable = (event) => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        acceptFile(new File([blob], `voice-${Date.now()}.webm`, { type: blob.type }));
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
      };
      recorderRef.current = recorder; recorder.start(); setRecording(true);
    } catch { setError("Microphone access was not granted. You can upload a recording instead."); }
  }

  async function analyze() {
    setError("");
    const patientAge = Number(age);
    if (!file) { setError("Upload or record a voice sample first."); return; }
    if (!Number.isInteger(patientAge) || patientAge < 18 || patientAge > 120) { setError("Patient age must be between 18 and 120."); return; }
    setProcessing(true); setStep(1);
    try {
      const result = await submitVoiceAssessment(file, patientAge, task);
      setStep(2);
      saveAssessment("Voice", result.prediction, { features: result.extracted_features, transcript: result.transcript });
      setStep(3); router.push("/voice/results");
    } catch (reason) { setStep(0); setError(reason instanceof Error ? reason.message : "Voice analysis failed."); }
    finally { setProcessing(false); }
  }

  return <AppShell><PageHeader title="Neurological Voice Assessment" description="Record or upload a standardized speech sample for structured research feature extraction and model assessment." />
    <div className="mb-7 grid grid-cols-4 gap-2">{steps.map((label,index)=><div key={label} className="text-center"><div className={`mx-auto grid size-9 place-items-center rounded-full border-2 text-sm font-bold ${index <= step ? "border-blue-700 bg-blue-700 text-white" : "border-slate-300 bg-white text-slate-400"}`}>{index+1}</div><div className={`mt-2 text-xs font-semibold sm:text-sm ${index <= step ? "text-blue-700" : "text-slate-400"}`}>{label}</div></div>)}</div>
    <Card><CardContent className="grid gap-7 p-5 sm:p-7 lg:grid-cols-[300px_1fr]">
      <div className="space-y-5 border-b border-slate-200 pb-7 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-7"><label className="block"><span className="mb-2 block text-sm font-semibold">Task Selection</span><select value={task} onChange={(e)=>setTask(e.target.value)} className="h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm"><option value="reading">Reading</option><option value="picture_description">Picture Description</option><option value="monologue">Monologue</option><option value="sustained_vowel">Sustained Vowel</option></select></label><label className="block"><span className="mb-2 block text-sm font-semibold">Patient Age</span><Input type="number" min={18} max={120} value={age} onChange={(e)=>setAge(e.target.value)} /></label><p className="text-sm leading-6 text-slate-500">Audio is sent to the configured Gemini model through FastAPI. The API key remains on the server.</p></div>
      <div className="flex flex-col items-center justify-center rounded-lg border border-slate-200 bg-slate-50 p-5 sm:p-8"><blockquote className="max-w-2xl text-center text-base italic leading-7 text-slate-600 sm:text-lg">“{passage}”</blockquote>
        {recording && <div className="mt-8 flex h-14 items-center gap-1" aria-label="Recording in progress">{Array.from({length:24}).map((_,index)=><span key={index} className="wave-bar h-10 w-1 rounded-full bg-blue-600" style={{animationDelay:`${index*45}ms`}} />)}</div>}
        {!file ? <div className="mt-8 flex flex-wrap justify-center gap-3"><Button size="lg" onClick={()=>inputRef.current?.click()}><UploadCloud className="size-5" />Upload audio</Button><Button type="button" variant={recording ? "danger" : "outline"} size="lg" onClick={toggleRecording}>{recording ? <><Square className="size-5" />Stop recording</> : <><Mic className="size-5" />Record sample</>}</Button></div> : <div className="mt-8 flex w-full max-w-xl items-center gap-3 rounded-lg border border-blue-200 bg-blue-50 p-4"><FileAudio className="size-6 shrink-0 text-blue-700" /><div className="min-w-0 flex-1"><div className="truncate text-sm font-semibold">{file.name}</div><div className="text-xs text-slate-500">{(file.size/1024/1024).toFixed(2)} MB</div></div><Button variant="ghost" size="icon" onClick={()=>setFile(null)} aria-label="Remove audio"><X className="size-5" /></Button></div>}
        <input ref={inputRef} className="hidden" type="file" accept="audio/wav,audio/mpeg,audio/mp4,audio/x-m4a,audio/ogg,audio/webm" onChange={(event)=>{const selected=event.target.files?.[0]; if(selected) acceptFile(selected);}} />
        <div className="mt-4 text-center text-xs leading-5 text-slate-500">WAV, MP3, M4A, OGG, or WebM. Maximum 18 MB.</div>
        {error && <div role="alert" className="mt-5 w-full max-w-xl rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">{error}</div>}
        <Button className="mt-6 min-w-56" size="lg" disabled={!file || processing || recording} onClick={analyze}>{processing ? <><Loader2 className="size-5 animate-spin" />Analyzing audio</> : "Analyze Voice Sample"}</Button>
      </div>
    </CardContent></Card>
  </AppShell>;
}
