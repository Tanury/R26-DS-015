"use client";

import { useState, useCallback, useRef } from "react";
import {
  analyzeMRI,
  preprocessDICOM,
  analyzeDatPipeline,
  analyzePdMRI,
  analyzeMsMRI,
  analyzeMsOct,
} from "@/lib/neuroimaging-api";
// ── API ─────────────────────────────────────────────────────────────────────
// Extracted to lib/neuroimaging-api.ts (env-based backend URL, matching the
// pattern used by the EEG branch's lib/eeg-api.ts) instead of living inline
// with a hardcoded localhost URL.


// Shared chrome (matches the rest of the platform: indigo accent, white
// cards on a light gray page, neutral slate text). Only `accent` /
// `accentStrong` differ per disease, used sparingly for score bars and
// the small "ACTIVE" indicators — everything else stays consistent.
const CHROME = {
  bg: "#F5F7FB", sidebar: "#FFFFFF", cardBg: "#FFFFFF",
  border: "#E5E8EF", borderLight: "#C7D9FB",
  text: "#111827", textMuted: "#4B5563", textDim: "#6B7280",
  headerBg: "#2F5FDC", shadow: "rgba(16,24,40,0.08)",
};

const THEMES = {
  alzheimers: {
    ...CHROME,
    id: "alzheimers", name: "Alzheimer's Disease", short: "Alzheimer's",
    ribbon: "#2F5FDC", accent: "#2F5FDC", accentStrong: "#1E40AF",
    accentLight: "#E8EFFE", accentMid: "#A8C0F5", barBg: "#F0F4FE",
  },
  parkinsons: {
    ...CHROME,
    id: "parkinsons", name: "Parkinson's Disease", short: "Parkinson's",
    ribbon: "#2F5FDC", accent: "#2F5FDC", accentStrong: "#1E40AF",
    accentLight: "#E8EFFE", accentMid: "#A8C0F5", barBg: "#F0F4FE",
  },
  ms: {
    ...CHROME,
    id: "ms", name: "Multiple Sclerosis", short: "Mult. Sclerosis",
    ribbon: "#2F5FDC", accent: "#2F5FDC", accentStrong: "#1E40AF",
    accentLight: "#E8EFFE", accentMid: "#A8C0F5", barBg: "#F0F4FE",
  },
};

const CLASS_CONFIG = {
  AD:  { label: "Alzheimer's Disease",      risk: "HIGH" },
  MCI: { label: "Mild Cognitive Impairment", risk: "MODERATE" },
  CN:  { label: "Cognitively Normal",        risk: "LOW" },
  PD:  { label: "Parkinson's Disease",       risk: "HIGH" },
  HC:  { label: "Healthy Control",           risk: "LOW" },
  MS:  { label: "Multiple Sclerosis",        risk: "HIGH" },
};

const AD_STEP_COLORS = ["#64748B","#3B82F6","#8B5CF6","#10B981"];
const AD_STEP_NAMES  = ["DICOM → NIfTI","Skull Strip","Registration","Bias Field Correction"];
const PD_STEP_COLORS = ["#3B82F6","#8B5CF6","#10B981","#F59E0B"];
const PD_STEP_NAMES  = ["DICOM → NIfTI","Slice Extraction","SBR Normalisation","2D-CNN Inference"];
const MS_OCT_STEP_COLORS = ["#3B82F6","#D35400","#F59E0B","#10B981"];
const MS_OCT_STEP_NAMES  = ["Parse .vol / .mat","Extract Central B-scans","Compute Layer Thickness","Thickness Encoder → Predict"];

// ── Icons ─────────────────────────────────────────────────────────────────────
// Plain outline icons, neutral across all three diseases — matches the
// rest of the platform's flat sidebar-icon style rather than illustrative
// per-disease symbols.
function BrainIcon({ size = 28, color = "#2F5FDC" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none">
      <path d="M9 6c-2.2 0-4 1.8-4 4 0 .7.2 1.4.5 2-.9.7-1.5 1.8-1.5 3 0 1.5.9 2.8 2.2 3.4-.1.4-.2.7-.2 1.1 0 2.2 1.8 4 4 4h.5V6H9Z"
        stroke={color} strokeWidth="1.6" strokeLinejoin="round"/>
      <path d="M19 6c2.2 0 4 1.8 4 4 0 .7-.2 1.4-.5 2 .9.7 1.5 1.8 1.5 3 0 1.5-.9 2.8-2.2 3.4.1.4.2.7.2 1.1 0 2.2-1.8 4-4 4h-.5V6H19Z"
        stroke={color} strokeWidth="1.6" strokeLinejoin="round"/>
    </svg>
  );
}

function DiseaseIcon({ theme, size = 28 }) {
  return <BrainIcon size={size} color={theme.accentStrong}/>;
}

function RibbonSVG({ color, size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" fill="none">
      <path d="M9 2 C6.5 2 4.5 4 4.5 6 C4.5 9 9 11 9 11 C9 11 13.5 9 13.5 6 C13.5 4 11.5 2 9 2Z" fill={color}/>
      <path d="M9 11 L6.5 17 L9 14 L11.5 17 Z" fill={color} opacity="0.75"/>
    </svg>
  );
}

// ── Card 3D ───────────────────────────────────────────────────────────────────
function Card3D({ children, theme, style = {}, hover = true }) {
  const [hov, setHov] = useState(false);
  return (
    <div
      onMouseEnter={() => hover && setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: theme.cardBg, border: `1px solid ${theme.border}`, borderRadius: 14,
        boxShadow: hov
          ? `0 12px 32px ${theme.shadow},0 2px 8px rgba(0,0,0,0.08),inset 0 1px 0 rgba(255,255,255,0.9)`
          : `0 4px 16px ${theme.shadow},0 1px 4px rgba(0,0,0,0.05),inset 0 1px 0 rgba(255,255,255,0.9)`,
        transform: hov ? "translateY(-3px)" : "translateY(0)",
        transition: "all 0.25s cubic-bezier(0.34,1.56,0.64,1)", ...style,
      }}
    >{children}</div>
  );
}

// ── Pipeline step card ─────────────────────────────────────────────────────────
function PipelineStepCard({ step, theme, stepColors }) {
  const color  = stepColors[step.step] || "#888";
  const status = step.success ? "done" : "error";
  return (
    <Card3D theme={theme} style={{ overflow: "hidden" }}>
      <div style={{ padding: "12px 14px 10px", borderBottom: `1px solid ${theme.border}`, display: "flex", alignItems: "center", justifyContent: "space-between", background: `linear-gradient(135deg,${color}12,${color}06)` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: color, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 12, fontWeight: 800, boxShadow: `0 3px 8px ${color}55` }}>
            {step.step}
          </div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: theme.text }}>{step.name}</div>
            <div style={{ fontSize: 9, color: theme.textDim }}>{step.elapsed_s ? `${step.elapsed_s}s` : "—"}</div>
          </div>
        </div>
        <div style={{ padding: "3px 10px", borderRadius: 12, fontSize: 9, fontWeight: 700, letterSpacing: 0.8, background: status === "done" ? "#D1FAE5" : "#FEE2E2", color: status === "done" ? "#065F46" : "#991B1B", border: `1px solid ${status === "done" ? "#6EE7B7" : "#FECACA"}` }}>
          {status === "done" ? "✓ DONE" : "✗ ERROR"}
        </div>
      </div>
      <div style={{ padding: "8px 14px 10px", fontSize: 10, color: theme.textDim, lineHeight: 1.5 }}>
        {step.description}
        {step.error && <div style={{ marginTop: 4, color: "#DC2626", fontSize: 9 }}>{step.error.slice(0,80)}</div>}
      </div>
      {step.slices && (
        <div style={{ padding: "0 10px 12px", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
          {["axial","coronal","sagittal"].map(view => (
            <div key={view} style={{ position: "relative" }}>
              {step.slices[view]
                ? <img src={`data:image/png;base64,${step.slices[view]}`} alt={view} style={{ width:"100%",aspectRatio:"1",borderRadius:8,objectFit:"cover",border:`1px solid ${theme.border}`,boxShadow:"0 2px 6px rgba(0,0,0,0.12)",display:"block" }}/>
                : <div style={{ width:"100%",aspectRatio:"1",borderRadius:8,background:"#0a0a0a",border:`1px solid ${theme.border}`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,color:"#555" }}>{view}</div>
              }
              <div style={{ position:"absolute",bottom:4,left:0,right:0,textAlign:"center",fontSize:8,color:"rgba(255,255,255,0.7)",textTransform:"uppercase",letterSpacing:0.5,textShadow:"0 1px 2px rgba(0,0,0,0.8)" }}>{view}</div>
            </div>
          ))}
        </div>
      )}
    </Card3D>
  );
}

// ── Thickness panel (OCT-specific) ──────────────────────────────────────────────
function ThicknessPanel({ thickness, theme }) {
  if (!thickness || !thickness.length) return null;
  const maxVal = Math.max(...thickness.flatMap(l => [l.value_um, l.hc_mean, l.ms_mean].filter(v => v != null))) * 1.15;
  return (
    <div style={{ marginTop:14,paddingTop:14,borderTop:`1px solid ${theme.border}` }}>
      <div style={{ fontSize:10,color:theme.textDim,letterSpacing:1.5,textTransform:"uppercase",fontWeight:700,marginBottom:10 }}>
        Retinal Layer Thickness · central B-scans
      </div>
      {thickness.map(l => (
        <div key={l.layer} style={{ marginBottom:11 }}>
          <div style={{ display:"flex",justifyContent:"space-between",marginBottom:3 }}>
            <span style={{ fontSize:11,color:theme.textMuted,fontWeight:600 }}>{l.layer}</span>
            <span style={{ fontSize:11,color:theme.text,fontWeight:700 }}>{l.value_um?.toFixed(1)} µm</span>
          </div>
          <div style={{ position:"relative",height:10,background:theme.barBg,borderRadius:5,border:`1px solid ${theme.border}` }}>
            <div style={{ position:"absolute",left:0,top:0,bottom:0,width:`${Math.min(100,(l.value_um/maxVal)*100)}%`,background:theme.accent,borderRadius:5,opacity:0.85,transition:"width 0.8s cubic-bezier(0.16,1,0.3,1)" }}/>
            {l.hc_mean!=null && (
              <div title={`Healthy avg: ${l.hc_mean.toFixed(1)}µm`} style={{ position:"absolute",left:`${Math.min(100,(l.hc_mean/maxVal)*100)}%`,top:-2,bottom:-2,width:2,background:"#16A34A" }}/>
            )}
            {l.ms_mean!=null && (
              <div title={`MS avg: ${l.ms_mean.toFixed(1)}µm`} style={{ position:"absolute",left:`${Math.min(100,(l.ms_mean/maxVal)*100)}%`,top:-2,bottom:-2,width:2,background:"#DC2626" }}/>
            )}
          </div>
        </div>
      ))}
      <div style={{ display:"flex",gap:14,marginTop:8,fontSize:9,color:theme.textDim }}>
        <span><span style={{display:"inline-block",width:8,height:8,background:theme.accent,borderRadius:2,marginRight:4,verticalAlign:"middle"}}/>This upload</span>
        <span><span style={{display:"inline-block",width:2,height:10,background:"#16A34A",marginRight:4,verticalAlign:"middle"}}/>Healthy avg</span>
        <span><span style={{display:"inline-block",width:2,height:10,background:"#DC2626",marginRight:4,verticalAlign:"middle"}}/>MS avg</span>
      </div>
    </div>
  );
}

// ── Region volume panel (MRI-specific) ──────────────────────────────────────────
function RegionVolumePanel({ regions, theme }) {
  if (!regions || !regions.length) return null;
  const maxVal = Math.max(...regions.flatMap(r => [r.value_frac, r.hc_mean, r.ms_mean].filter(v => v != null))) * 1.15;
  const label = (name) => name.replace(/_/g, " ");
  return (
    <div style={{ marginTop:14,paddingTop:14,borderTop:`1px solid ${theme.border}` }}>
      <div style={{ fontSize:10,color:theme.textDim,letterSpacing:1.5,textTransform:"uppercase",fontWeight:700,marginBottom:10 }}>
        Region Volumes · % of total brain (MindGlide)
      </div>
      {regions.map(r => (
        <div key={r.region} style={{ marginBottom:11 }}>
          <div style={{ display:"flex",justifyContent:"space-between",marginBottom:3 }}>
            <span style={{ fontSize:11,color:theme.textMuted,fontWeight:600 }}>{label(r.region)}</span>
            <span style={{ fontSize:11,color:theme.text,fontWeight:700 }}>{(r.value_frac*100).toFixed(2)}%</span>
          </div>
          <div style={{ position:"relative",height:10,background:theme.barBg,borderRadius:5,border:`1px solid ${theme.border}` }}>
            <div style={{ position:"absolute",left:0,top:0,bottom:0,width:`${Math.min(100,(r.value_frac/maxVal)*100)}%`,background:theme.accent,borderRadius:5,opacity:0.85,transition:"width 0.8s cubic-bezier(0.16,1,0.3,1)" }}/>
            {r.hc_mean!=null && (
              <div title={`Healthy avg: ${(r.hc_mean*100).toFixed(2)}%`} style={{ position:"absolute",left:`${Math.min(100,(r.hc_mean/maxVal)*100)}%`,top:-2,bottom:-2,width:2,background:"#16A34A" }}/>
            )}
            {r.ms_mean!=null && (
              <div title={`MS avg: ${(r.ms_mean*100).toFixed(2)}%`} style={{ position:"absolute",left:`${Math.min(100,(r.ms_mean/maxVal)*100)}%`,top:-2,bottom:-2,width:2,background:"#DC2626" }}/>
            )}
          </div>
        </div>
      ))}
      <div style={{ display:"flex",gap:14,marginTop:8,fontSize:9,color:theme.textDim }}>
        <span><span style={{display:"inline-block",width:8,height:8,background:theme.accent,borderRadius:2,marginRight:4,verticalAlign:"middle"}}/>This upload</span>
        <span><span style={{display:"inline-block",width:2,height:10,background:"#16A34A",marginRight:4,verticalAlign:"middle"}}/>Healthy avg</span>
        <span><span style={{display:"inline-block",width:2,height:10,background:"#DC2626",marginRight:4,verticalAlign:"middle"}}/>MS avg</span>
      </div>
    </div>
  );
}

// ── Result card ────────────────────────────────────────────────────────────────
function ResultCard({ prediction, theme }) {
  const cfg = CLASS_CONFIG[prediction.prediction] || {};
  const rc  = { HIGH:{ bg:"#FEE2E2",text:"#991B1B",border:"#FECACA" }, MODERATE:{ bg:"#FEF9C3",text:"#854D0E",border:"#FDE68A" }, LOW:{ bg:"#DCFCE7",text:"#166534",border:"#BBF7D0" } }[prediction.risk_level] || { bg:"#DCFCE7",text:"#166534",border:"#BBF7D0" };
  return (
    <Card3D theme={theme} style={{ padding:"18px 20px" }} hover={false}>
      <div style={{ fontSize:10,color:theme.textDim,letterSpacing:2,textTransform:"uppercase",fontWeight:700,marginBottom:14 }}>Final Prediction</div>
      <div style={{ display:"flex",alignItems:"flex-start",justifyContent:"space-between",marginBottom:14 }}>
        <div>
          <div style={{ fontSize:40,fontWeight:900,color:theme.accentStrong,lineHeight:1,marginBottom:4 }}>{prediction.prediction}</div>
          <div style={{ fontSize:11,color:theme.textDim }}>{prediction.confidence?.toFixed(1)}% confidence</div>
        </div>
        <div style={{ padding:"6px 14px",borderRadius:20,background:rc.bg,border:`1px solid ${rc.border}`,display:"flex",alignItems:"center",gap:6 }}>
          <div style={{ width:7,height:7,borderRadius:"50%",background:rc.text }}/>
          <span style={{ fontSize:10,fontWeight:700,color:rc.text,letterSpacing:0.8 }}>{prediction.risk_level} RISK</span>
        </div>
      </div>
      <div style={{ fontSize:12,color:theme.textMuted,marginBottom:14 }}>{cfg.label}</div>
      {prediction.class_scores?.map((s,i) => (
        <div key={s.label} style={{ marginBottom:10 }}>
          <div style={{ display:"flex",justifyContent:"space-between",marginBottom:4 }}>
            <span style={{ fontSize:12,color:theme.textMuted }}>{s.label}</span>
            <span style={{ fontSize:12,fontWeight:i===0?700:400,color:i===0?theme.accentStrong:theme.textDim }}>{(s.confidence*100).toFixed(1)}%</span>
          </div>
          <div style={{ height:8,background:theme.barBg,borderRadius:4,overflow:"hidden",border:`1px solid ${theme.border}` }}>
            <div style={{ height:"100%",width:`${s.confidence*100}%`,background:i===0?theme.accent:theme.accentMid,borderRadius:4,opacity:i===0?1:0.45,transition:"width 1s cubic-bezier(0.16,1,0.3,1)" }}/>
          </div>
        </div>
      ))}
      {prediction.modality && (
        <div style={{ marginTop:10,padding:"6px 10px",borderRadius:6,background:theme.accentLight,border:`1px solid ${theme.border}`,display:"flex",justifyContent:"space-between" }}>
          <span style={{ fontSize:10,color:theme.textMuted }}>Modality</span>
          <span style={{ fontSize:10,fontWeight:700,color:theme.accentStrong }}>{prediction.modality}</span>
        </div>
      )}
      <ThicknessPanel thickness={prediction.thickness} theme={theme}/>
      <RegionVolumePanel regions={prediction.regions} theme={theme}/>
      {prediction.slice_image && (
        <div style={{ marginTop:14,paddingTop:14,borderTop:`1px solid ${theme.border}` }}>
          <div style={{ fontSize:10,color:theme.textDim,letterSpacing:1.5,textTransform:"uppercase",fontWeight:700,marginBottom:10 }}>
            Segmentation · Input vs MindGlide
          </div>
          <img src={prediction.slice_image} alt="Input MRI vs MindGlide segmentation overlay"
               style={{ width:"100%",borderRadius:8,border:`1px solid ${theme.border}`,display:"block" }}/>
          <div style={{ fontSize:9,color:theme.textDim,marginTop:6 }}>
            Automatically selected slice — the one with the most detected lesion tissue (or largest brain cross-section if none found).
          </div>
        </div>
      )}
      <div style={{ marginTop:10,padding:"10px 12px",borderRadius:8,background:theme.accentLight,border:`1px solid ${theme.border}` }}>
        <div style={{ display:"flex",justifyContent:"space-between",marginBottom:5 }}>
          <span style={{ fontSize:11,color:theme.textMuted }}>z_img embedding</span>
          <span style={{ fontSize:11,fontWeight:700,color:theme.accentStrong }}>256-d L2-normalised</span>
        </div>
        <div style={{ fontSize:10,color:theme.textDim }}>Ready for multimodal fusion engine →</div>
      </div>
      <div style={{ marginTop:12,padding:"8px 10px",borderRadius:6,background:"#FFFDE7",border:"1px solid #F9A825",fontSize:10,color:"#5D4037",lineHeight:1.5 }}>
        ⚠ Non-diagnostic output only · Not a substitute for clinical evaluation
      </div>
    </Card3D>
  );
}

// ── Upload zone ────────────────────────────────────────────────────────────────
function UploadZone({ onFile, loading, theme, mode }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();
  const drop = useCallback((e) => { e.preventDefault(); setDrag(false); const f=e.dataTransfer.files[0]; if(f) onFile(f); }, [onFile]);
  const accept   = mode==="dicom" ? ".zip" : mode==="dat" ? ".dcm,.ima,.img" : ".nii,.nii.gz";
  const label    = mode==="dicom" ? "Drop DICOM zip here" : mode==="dat" ? "Drop DaTscan DICOM here" : "Drop NIfTI file here";
  const sublabel = mode==="dicom" ? "ZIP of DICOM folder · Full pipeline + prediction"
                 : mode==="dat"   ? ".dcm · Reconstructed DaTscan SPECT"
                 :                  ".nii · .nii.gz · preprocessed brain MRI";
  return (
    <div onClick={()=>!loading&&ref.current?.click()} onDragOver={(e)=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={drop}
      style={{ border:`2px dashed ${drag?theme.accent:theme.borderLight}`,borderRadius:12,padding:"28px 16px",textAlign:"center",cursor:loading?"not-allowed":"pointer",background:drag?theme.accentLight:theme.barBg,transition:"all 0.2s",boxShadow:drag?`0 4px 16px ${theme.shadow}`:"none" }}>
      <input ref={ref} type="file" accept={accept} style={{display:"none"}} onChange={(e)=>{const f=e.target.files[0];if(f) onFile(f);}}/>
      <div style={{marginBottom:10}}><DiseaseIcon theme={theme} size={36}/></div>
      <div style={{color:theme.text,fontSize:13,fontWeight:600,marginBottom:4}}>{loading?"Processing…":label}</div>
      <div style={{color:theme.textDim,fontSize:11}}>{sublabel}</div>
    </div>
  );
}

// ── OCT dual-file upload zone ────────────────────────────────────────────────
function OctUploadZone({ onFiles, loading, theme }) {
  const [volFile, setVolFile] = useState(null);
  const [matFile, setMatFile] = useState(null);
  const volRef = useRef(); const matRef = useRef();
  const bothReady = volFile && matFile;

  const slot = (label, ext, f, setF, ref) => {
    const [drag, setDrag] = useState(false);
    const drop = (e) => { e.preventDefault(); setDrag(false); const file = e.dataTransfer.files[0]; if (file) setF(file); };
    return (
      <div onClick={()=>!loading&&ref.current?.click()}
        onDragOver={(e)=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={drop}
        style={{ border:`2px dashed ${drag?theme.accent:f?theme.borderLight:theme.border}`,borderRadius:10,padding:"14px 12px",textAlign:"center",cursor:loading?"not-allowed":"pointer",background:f?theme.accentLight:theme.barBg,transition:"all 0.2s",marginBottom:8 }}>
        <input ref={ref} type="file" accept={ext} style={{display:"none"}} onChange={(e)=>{const file=e.target.files[0];if(file) setF(file);}}/>
        <div style={{fontSize:11,fontWeight:700,color:f?theme.accentStrong:theme.text,marginBottom:2}}>{f ? `✓ ${label}` : label}</div>
        <div style={{fontSize:10,color:theme.textDim,wordBreak:"break-all"}}>{f ? f.name : `Drop ${ext} file here`}</div>
      </div>
    );
  };

  return (
    <div>
      {slot(".vol scan", ".vol", volFile, setVolFile, volRef)}
      {slot(".mat delineation", ".mat", matFile, setMatFile, matRef)}
      <button
        disabled={!bothReady || loading}
        onClick={() => onFiles(volFile, matFile)}
        style={{ width:"100%",padding:"10px 12px",borderRadius:8,border:"none",fontSize:12,fontWeight:700,cursor:bothReady&&!loading?"pointer":"not-allowed",background:bothReady&&!loading?theme.accent:theme.border,color:bothReady&&!loading?"#fff":theme.textDim,transition:"all 0.2s",marginTop:4 }}>
        {loading ? "Analysing…" : "Run OCT Analysis"}
      </button>
      <div style={{fontSize:9,color:theme.textDim,marginTop:6,lineHeight:1.5}}>
        Requires both the raw scan and its manual layer delineation — matching the format used to train and validate the model.
      </div>
    </div>
  );
}

// ── Mode toggle ────────────────────────────────────────────────────────────────
function ModeToggle({ mode, setMode, theme, isParkinsons, isMS }) {
  const modes = isMS
    ? [{ id:"oct",  label:"👁 OCT (.vol+.mat) → Predict" },{ id:"mri",label:"🧠 NIfTI → Predict" }]
    : isParkinsons
    ? [{ id:"dat",  label:"⚡ DaTscan → Pipeline" },{ id:"nifti",label:"🧠 NIfTI → Predict" }]
    : [{ id:"dicom",label:"⚡ DICOM → Pipeline"  },{ id:"nifti",label:"🧠 NIfTI → Predict" }];
  return (
    <div style={{display:"flex",flexDirection:"column",gap:6}}>
      {modes.map(m=>(
        <button key={m.id} onClick={()=>setMode(m.id)} style={{ padding:"9px 12px",borderRadius:8,fontSize:12,fontWeight:mode===m.id?700:400,border:`1px solid ${mode===m.id?theme.borderLight:theme.border}`,background:mode===m.id?theme.accentLight:"transparent",color:mode===m.id?theme.accentStrong:theme.textMuted,cursor:"pointer",transition:"all 0.2s",textAlign:"left",boxShadow:mode===m.id?`0 2px 8px ${theme.shadow}`:"none" }}>
          {m.label}
        </button>
      ))}
    </div>
  );
}

// ── Dashboard ──────────────────────────────────────────────────────────────────
export default function NeuroimagingPage() {
  const [active,     setActive]     = useState("alzheimers");
  const [mode,       setMode]       = useState("dicom");
  const [file,       setFile]       = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [pipeline,   setPipeline]   = useState(null);
  const [error,      setError]      = useState(null);
  const [activeStep, setActiveStep] = useState(-1);

  const t            = THEMES[active];
  const isParkinsons = active === "parkinsons";
  const isMS         = active === "ms";
  const stepColors   = isMS ? MS_OCT_STEP_COLORS : isParkinsons ? PD_STEP_COLORS : AD_STEP_COLORS;
  const stepNames    = isMS ? MS_OCT_STEP_NAMES  : isParkinsons ? PD_STEP_NAMES  : AD_STEP_NAMES;

  const switchDisease = (id) => {
    setActive(id); setFile(null); setResult(null); setPipeline(null);
    setError(null); setLoading(false);
    setMode(id === "parkinsons" ? "dat" : id === "ms" ? "oct" : "dicom");
  };

  const handleModeChange = (m) => {
    setMode(m); setFile(null); setResult(null); setPipeline(null); setError(null);
  };

  const handleFile = useCallback(async (f) => {
    setFile(f); setResult(null); setPipeline(null); setError(null);
    setLoading(true); setActiveStep(0);
    try {
      if (isParkinsons && mode === "dat") {
        // DaTscan pipeline — step-by-step
        const timer = setInterval(() => setActiveStep(s => Math.min(s+1, 3)), 1500);
        const data  = await analyzeDatPipeline(f);
        clearInterval(timer); setActiveStep(-1); setPipeline(data);
      } else if (isParkinsons && mode === "nifti") {
        const data = await analyzePdMRI(f);
        setResult(data);
      } else if (isMS && mode === "mri") {
        const data = await analyzeMsMRI(f);
        setResult(data);
      } else if (mode === "dicom") {
        const timer = setInterval(() => setActiveStep(s => Math.min(s+1, 3)), 8000);
        const data  = await preprocessDICOM(f);
        clearInterval(timer); setActiveStep(-1); setPipeline(data);
      } else {
        const data = await analyzeMRI(f);
        setResult(data);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false); setActiveStep(-1);
    }
  }, [mode, isParkinsons, isMS]);

  const handleOctFiles = useCallback(async (volFile, matFile) => {
    setFile(volFile); setResult(null); setPipeline(null); setError(null);
    setLoading(true); setActiveStep(0);
    const timer = setInterval(() => setActiveStep(s => Math.min(s+1, 3)), 900);
    try {
      const data = await analyzeMsOct(volFile, matFile);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      clearInterval(timer); setLoading(false); setActiveStep(-1);
    }
  }, []);

  const hasPipeline = pipeline && pipeline.pipeline;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        body{background:${t.bg};color:${t.text};font-family:'Inter',sans-serif;min-height:100vh;transition:background 0.3s;}
        @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
        ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.border};border-radius:3px}
      `}</style>

      <div style={{display:"flex",flexDirection:"column",minHeight:"100vh"}}>

        {/* Header */}
        <header style={{background:t.headerBg,padding:"0 28px"}}>
          <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"14px 0 10px"}}>
            <div style={{display:"flex",alignItems:"center",gap:12}}>
              <div style={{width:40,height:40,borderRadius:10,background:"rgba(255,255,255,0.2)",border:"1px solid rgba(255,255,255,0.35)",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 4px 12px rgba(0,0,0,0.15),inset 0 1px 0 rgba(255,255,255,0.3)"}}>
                <DiseaseIcon theme={t} size={24}/>
              </div>
              <div>
                <div style={{fontSize:17,fontWeight:700,color:"#fff",fontFamily:"'Inter',sans-serif",letterSpacing:-0.3}}>NeuroRisk AI</div>
                <div style={{fontSize:9,color:"rgba(255,255,255,0.7)",letterSpacing:1.5,textTransform:"uppercase"}}>R26-DS-015 · Adaptive Multimodal Neurological Risk Assessment</div>
              </div>
            </div>
            <div style={{display:"flex",gap:6}}>
              {Object.values(THEMES).map(d=>(
                <button key={d.id} onClick={()=>switchDisease(d.id)} style={{ padding:"7px 16px",borderRadius:20,border:active===d.id?"1.5px solid rgba(255,255,255,0.85)":"1.5px solid rgba(255,255,255,0.25)",background:active===d.id?"rgba(255,255,255,0.25)":"rgba(255,255,255,0.1)",color:"#fff",fontSize:12,fontWeight:active===d.id?700:400,cursor:"pointer",transition:"all 0.2s",display:"flex",alignItems:"center",gap:6,boxShadow:active===d.id?"0 2px 8px rgba(0,0,0,0.2)":"none" }}>
                  <RibbonSVG color={d.id==="parkinsons"?"#E0E0E0":"#fff"} size={13}/>
                  {d.short}
                </button>
              ))}
            </div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:10,padding:"8px 0 12px",borderTop:"1px solid rgba(255,255,255,0.15)"}}>
            <RibbonSVG color={t.id==="parkinsons"?"#E0E0E0":"rgba(255,255,255,0.9)"} size={15}/>
            <span style={{fontSize:12,fontWeight:600,color:"rgba(255,255,255,0.95)"}}>{t.name}</span>
            {isParkinsons&&<span style={{fontSize:10,background:"rgba(255,255,255,0.2)",color:"rgba(255,255,255,0.9)",padding:"2px 8px",borderRadius:10,fontWeight:600}}>DaTscan AUC 93.75% · Active</span>}
          </div>
        </header>

        {/* Body */}
        <div style={{display:"flex",flex:1}}>

          {/* Sidebar */}
          <aside style={{width:290,background:t.sidebar,borderRight:`1px solid ${t.border}`,padding:18,display:"flex",flexDirection:"column",gap:16,boxShadow:`2px 0 12px ${t.shadow}`}}>
            <div>
              <div style={{fontSize:9,color:t.textDim,letterSpacing:2,textTransform:"uppercase",marginBottom:8,fontWeight:700}}>Input Mode</div>
              <ModeToggle mode={mode} setMode={handleModeChange} theme={t} isParkinsons={isParkinsons} isMS={isMS}/>
            </div>
            <div>
              <div style={{fontSize:9,color:t.textDim,letterSpacing:2,textTransform:"uppercase",marginBottom:8,fontWeight:700}}>
                {mode==="dat"?"Upload · DaTscan DICOM":mode==="dicom"?"Upload · DICOM Zip":mode==="oct"?"Upload · OCT Scan + Delineation":"Upload · NIfTI"}
              </div>
              {isMS && mode==="oct"
                ? <OctUploadZone onFiles={handleOctFiles} loading={loading} theme={t}/>
                : <UploadZone onFile={handleFile} loading={loading} theme={t} mode={mode}/>
              }
            </div>
            {file&&(
              <Card3D theme={t} style={{padding:"10px 12px"}} hover={false}>
                <div style={{fontSize:9,color:t.textDim,marginBottom:4,letterSpacing:1.5,textTransform:"uppercase",fontWeight:700}}>Selected</div>
                <div style={{fontSize:11,color:t.text,fontWeight:600,wordBreak:"break-all"}}>{file.name}</div>
                <div style={{fontSize:10,color:t.textDim,marginTop:3}}>{(file.size/1048576).toFixed(1)} MB</div>
              </Card3D>
            )}
            {loading&&(
              <div>
                <div style={{fontSize:9,color:t.textDim,letterSpacing:2,textTransform:"uppercase",marginBottom:8,fontWeight:700}}>Pipeline Progress</div>
                {stepNames.map((name,i)=>(
                  <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"6px 0",borderBottom:`1px solid ${t.border}`,fontSize:11}}>
                    <div style={{width:20,height:20,borderRadius:6,background:i<activeStep?"#D1FAE5":i===activeStep?stepColors[i]:t.barBg,display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,fontWeight:800,color:i<activeStep?"#065F46":i===activeStep?"#fff":t.textDim,animation:i===activeStep?"pulse 1s ease infinite":"none"}}>
                      {i<activeStep?"✓":i}
                    </div>
                    <span style={{color:i<=activeStep?t.text:t.textDim,fontWeight:i===activeStep?600:400}}>{name}</span>
                  </div>
                ))}
              </div>
            )}
            <div>
              <div style={{fontSize:9,color:t.textDim,letterSpacing:2,textTransform:"uppercase",marginBottom:8,fontWeight:700}}>Analysis Target</div>
              {Object.values(THEMES).map(d=>(
                <div key={d.id} onClick={()=>switchDisease(d.id)} style={{ padding:"9px 12px",marginBottom:4,borderRadius:8,background:active===d.id?d.accentLight:"transparent",border:`1px solid ${active===d.id?d.borderLight:"transparent"}`,display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",transition:"all 0.2s",boxShadow:active===d.id?`0 2px 8px ${d.shadow}`:"none" }}>
                  <div style={{display:"flex",alignItems:"center",gap:8}}>
                    <RibbonSVG color={d.ribbon} size={14}/>
                    <span style={{fontSize:12,color:active===d.id?d.accentStrong:t.textDim,fontWeight:active===d.id?700:400}}>{d.short}</span>
                  </div>
                  <span style={{fontSize:9,color:active===d.id?d.accent:t.textDim,fontWeight:700,letterSpacing:1}}>ACTIVE</span>
                </div>
              ))}
            </div>
          </aside>

          {/* Main */}
          <main style={{flex:1,padding:24,background:t.bg,overflowY:"auto"}}>
            {!loading&&!hasPipeline&&!result&&!error&&(
              <div style={{display:"flex",alignItems:"center",justifyContent:"center",height:"100%",minHeight:400}}>
                <div style={{textAlign:"center"}}>
                  <div style={{animation:"float 3s ease-in-out infinite",marginBottom:20}}><DiseaseIcon theme={t} size={80}/></div>
                  <div style={{fontSize:18,color:t.text,fontWeight:700,marginBottom:8}}>
                    {mode==="dat"?"Upload a DaTscan DICOM to begin":mode==="dicom"?"Upload a DICOM zip to begin":mode==="oct"?"Upload a .vol scan and .mat delineation to begin":"Upload a NIfTI file to begin"}
                  </div>
                  <div style={{fontSize:12,color:t.textMuted,marginBottom:4}}>{t.name} · {mode==="dat"?"DaTscan SPECT Analysis":mode==="oct"?"Retinal OCT Thickness Analysis":"Brain MRI Analysis"}</div>
                  {mode==="oct"&&(
                    <div style={{marginTop:16,padding:"12px 20px",borderRadius:8,background:t.accentLight,border:`1px solid ${t.border}`,display:"inline-block",textAlign:"left"}}>
                      <div style={{fontSize:11,color:t.textMuted,marginBottom:6,fontWeight:600}}>OCT Thickness Pipeline</div>
                      {MS_OCT_STEP_NAMES.map((name,i)=>(
                        <div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                          <div style={{width:18,height:18,borderRadius:5,background:MS_OCT_STEP_COLORS[i],display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",fontSize:9,fontWeight:800}}>{i}</div>
                          <span style={{fontSize:11,color:t.textMuted}}>{name}</span>
                        </div>
                      ))}
                      <div style={{marginTop:8,fontSize:10,color:t.textDim}}>Model: Random Forest reference — 65.7% accuracy · 0.745 ROC-AUC (LOOCV)</div>
                    </div>
                  )}
                  {mode==="dat"&&(
                    <div style={{marginTop:16,padding:"12px 20px",borderRadius:8,background:t.accentLight,border:`1px solid ${t.border}`,display:"inline-block",textAlign:"left"}}>
                      <div style={{fontSize:11,color:t.textMuted,marginBottom:6,fontWeight:600}}>DaTscan Pipeline</div>
                      {PD_STEP_NAMES.map((name,i)=>(
                        <div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                          <div style={{width:18,height:18,borderRadius:5,background:PD_STEP_COLORS[i],display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",fontSize:9,fontWeight:800}}>{i}</div>
                          <span style={{fontSize:11,color:t.textMuted}}>{name}</span>
                        </div>
                      ))}
                      <div style={{marginTop:8,fontSize:10,color:t.textDim}}>Model: AUC 93.75% · Sensitivity 93.48% · Specificity 80.77%</div>
                    </div>
                  )}
                  {mode==="dicom"&&(
                    <div style={{marginTop:16,padding:"10px 20px",borderRadius:8,background:t.accentLight,border:`1px solid ${t.border}`,display:"inline-block"}}>
                      <div style={{fontSize:11,color:t.textMuted}}>Full pipeline: DICOM → NIfTI → Registration → Skull Strip → Bias Correct → Prediction</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {loading&&(
              <div style={{display:"flex",alignItems:"center",justifyContent:"center",height:"100%",minHeight:400}}>
                <div style={{textAlign:"center"}}>
                  <div style={{width:56,height:56,margin:"0 auto 20px",border:`4px solid ${t.accentLight}`,borderTop:`4px solid ${t.accent}`,borderRadius:"50%",animation:"spin 0.9s linear infinite",boxShadow:`0 0 20px ${t.shadow}`}}/>
                  <div style={{fontSize:15,color:t.text,fontWeight:700,marginBottom:6}}>
                    {mode==="dat"?`Running Step ${activeStep}: ${PD_STEP_NAMES[activeStep]||"Processing"}…`:mode==="dicom"?`Running Step ${activeStep}: ${AD_STEP_NAMES[activeStep]||"Processing"}…`:mode==="oct"?`Running Step ${activeStep}: ${MS_OCT_STEP_NAMES[activeStep]||"Processing"}…`:"Analysing…"}
                  </div>
                  <div style={{fontSize:11,color:t.textDim}}>
                    {mode==="dat"?"DaTscan SPECT · SBR Normalisation · DaTEncoder 2D-CNN":mode==="dicom"?"FSL + ANTs preprocessing pipeline":mode==="oct"?"Central-window thickness · ThicknessEncoder · 256-d z_img":"VisionEncoder · 3D ResNet-18 · 256-d z_img"}
                  </div>
                </div>
              </div>
            )}

            {error&&(
              <Card3D theme={t} style={{padding:"20px 24px",maxWidth:600,margin:"40px auto"}} hover={false}>
                <div style={{fontSize:14,fontWeight:700,color:"#991B1B",marginBottom:8}}>✕ Processing Error</div>
                <div style={{fontSize:12,color:"#7F1D1D",lineHeight:1.6}}>{error}</div>
                <button onClick={()=>{setError(null);setFile(null);}} style={{marginTop:14,padding:"7px 16px",borderRadius:8,border:"none",background:t.accent,color:"#fff",fontSize:12,fontWeight:600,cursor:"pointer"}}>Try Again</button>
              </Card3D>
            )}

            {result&&!hasPipeline&&(
              <div style={{maxWidth:520,margin:"0 auto",animation:"fadeUp 0.5s ease both"}}>
                <ResultCard prediction={result} theme={t}/>
              </div>
            )}

            {hasPipeline&&(
              <div style={{animation:"fadeUp 0.5s ease both"}}>
                <div style={{marginBottom:20,display:"flex",alignItems:"center",justifyContent:"space-between"}}>
                  <div>
                    <div style={{fontSize:16,fontWeight:800,color:t.text,marginBottom:2}}>
                      {isParkinsons?"DaTscan Processing Pipeline":"Preprocessing Pipeline"}
                    </div>
                    <div style={{fontSize:11,color:t.textDim}}>
                      {pipeline.subject_id} · {pipeline.total_elapsed_s}s total
                      {!pipeline.pipeline_success&&<span style={{color:"#DC2626",marginLeft:8}}>— Pipeline error</span>}
                    </div>
                  </div>
                  <div style={{padding:"5px 14px",borderRadius:20,fontSize:11,fontWeight:700,background:pipeline.pipeline_success?"#D1FAE5":"#FEE2E2",color:pipeline.pipeline_success?"#065F46":"#991B1B",border:`1px solid ${pipeline.pipeline_success?"#6EE7B7":"#FECACA"}`}}>
                    {pipeline.pipeline_success?"✓ Pipeline Complete":"✗ Pipeline Failed"}
                  </div>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:20}}>
                  {pipeline.pipeline.map(step=>(
                    <PipelineStepCard key={step.step} step={step} theme={t} stepColors={stepColors}/>
                  ))}
                </div>
                {pipeline.prediction&&!pipeline.prediction.error&&(
                  <div>
                    <div style={{fontSize:12,fontWeight:700,color:t.textMuted,letterSpacing:1,textTransform:"uppercase",marginBottom:12}}>→ Inference Result</div>
                    <ResultCard prediction={pipeline.prediction} theme={t}/>
                  </div>
                )}
                {pipeline.prediction?.error&&(
                  <Card3D theme={t} style={{padding:"14px 18px"}} hover={false}>
                    <div style={{fontSize:12,color:"#DC2626"}}>Inference failed: {pipeline.prediction.error}</div>
                  </Card3D>
                )}
              </div>
            )}
          </main>
        </div>

        {/* Footer */}
        <footer style={{padding:"10px 28px",background:t.sidebar,borderTop:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",fontSize:10,color:t.textDim,boxShadow:`0 -2px 8px ${t.shadow}`}}>
          <span>R26-DS-015 | Dissanayaka D.M.T.M | IT22203762</span>
          <span>NON-DIAGNOSTIC DECISION-SUPPORT OUTPUT ONLY</span>
        </footer>
      </div>
    </>
  );
}