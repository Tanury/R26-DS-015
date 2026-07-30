import { useState, useCallback, useRef } from "react";

// ─── Themes ──────────────────────────────────────────────────────────────────
const THEMES = {
  alzheimers: {
    id: "alzheimers", name: "Alzheimer's Disease", short: "Alzheimer's",
    ribbon: "#7B4FA6", accent: "#9B6DC0", accentStrong: "#6D28D9",
    accentLight: "#EDE5F5", accentMid: "#C9A8E0",
    bg: "#F5F0FA", sidebar: "#EEE5F7", cardBg: "#FFFFFF",
    border: "#D4BEE8", borderLight: "#C4A8DC",
    text: "#2D1B4E", textMuted: "#6B4E8A", textDim: "#9B7AB8",
    decorSymbol: "✿", headerBg: "#7C3AED",
    shadow: "rgba(124,58,237,0.15)",
  },
  parkinsons: {
    id: "parkinsons", name: "Parkinson's Disease", short: "Parkinson's",
    ribbon: "#C0C0C0", accent: "#C0392B", accentStrong: "#991B1B",
    accentLight: "#FDECEA", accentMid: "#F5A89E",
    bg: "#FDF5F5", sidebar: "#FAE8E8", cardBg: "#FFFFFF",
    border: "#F0BABA", borderLight: "#E89090",
    text: "#4A0E0E", textMuted: "#8B3028", textDim: "#B07070",
    //barBg: "#FDF0F0", tagline: "Silver ribbon · Red tulip flower",
    decorSymbol: "🌷", headerBg: "#DC2626",
    shadow: "rgba(220,38,38,0.15)",
  },
  ms: {
    id: "ms", name: "Multiple Sclerosis", short: "Mult. Sclerosis",
    ribbon: "#E67E22", accent: "#D35400", accentStrong: "#9A3412",
    accentLight: "#FEF0E7", accentMid: "#F5B880",
    bg: "#FFFAF5", sidebar: "#FEF0E0", cardBg: "#FFFFFF",
    border: "#F0C090", borderLight: "#E8A060",
    text: "#4A1A00", textMuted: "#8B4513", textDim: "#B07840",
    //barBg: "#FEF5EC", tagline: "Orange ribbon · Orange butterfly",
    decorSymbol: "ʚїɞ", headerBg: "#EA580C",
    shadow: "rgba(234,88,12,0.15)",
  },
};

const CLASS_CONFIG = {
  AD:  { label: "Alzheimer's Disease",       risk: "HIGH" },
  MCI: { label: "Mild Cognitive Impairment",  risk: "MODERATE" },
  CN:  { label: "Cognitively Normal",         risk: "LOW" },
  PD:  { label: "Parkinson's Disease",        risk: "HIGH" },
  HC:  { label: "Healthy Control",            risk: "LOW" },
  MS:  { label: "Multiple Sclerosis",         risk: "HIGH" },
};

const STEP_COLORS = ["#64748B", "#3B82F6", "#8B5CF6", "#10B981"];
const STEP_NAMES  = ["DICOM → NIfTI", "Skull Strip", "Registration"," Bias Field Correction"];

// ─── API ─────────────────────────────────────────────────────────────────────
async function analyzeMRI(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("http://localhost:8000/api/analyze/mri", { method: "POST", body: form });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || "Analysis failed"); }
  return res.json();
}

async function preprocessDICOM(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("http://localhost:8000/api/preprocess/dicom", { method: "POST", body: form });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || "Preprocessing failed"); }
  return res.json();
}

// ─── SVG icons ───────────────────────────────────────────────────────────────
function ForgetMeNot({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="3" fill="#FCD34D"/>
      {[0,60,120,180,240,300].map((deg, i) => (
        <ellipse key={i} cx="14" cy="14" rx="3" ry="6.5" fill="#9B6DC0" opacity="0.85"
          transform={`rotate(${deg} 14 14) translate(0 -5)`}/>
      ))}
    </svg>
  );
}

function RedTulip({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none">
      <path d="M14 26 C14 26 14 14 14 11" stroke="#22C55E" strokeWidth="2" strokeLinecap="round"/>
      <path d="M14 11 C12 9 10 7 11 4 C12 6 13 7 14 7 C15 7 16 6 17 4 C18 7 16 9 14 11Z" fill="#DC2626"/>
      <ellipse cx="14" cy="6" rx="3.5" ry="4" fill="#EF4444" opacity="0.9"/>
    </svg>
  );
}

function Butterfly({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none">
      <path d="M14 14 C9 9 4 7 4 12 C4 17 9 15 14 14Z" fill="#F97316" opacity="0.9"/>
      <path d="M14 14 C19 9 24 7 24 12 C24 17 19 15 14 14Z" fill="#F97316" opacity="0.9"/>
      <path d="M14 14 C10 17 5 18 7 22 C9 24 12 21 14 14Z" fill="#FDBA74" opacity="0.8"/>
      <path d="M14 14 C18 17 23 18 21 22 C19 24 16 21 14 14Z" fill="#FDBA74" opacity="0.8"/>
      <ellipse cx="14" cy="14" rx="1.5" ry="3" fill="#431407"/>
    </svg>
  );
}

function DiseaseIcon({ theme, size = 28 }) {
  if (theme.id === "alzheimers") return <ForgetMeNot size={size}/>;
  if (theme.id === "parkinsons") return <RedTulip size={size}/>;
  return <Butterfly size={size}/>;
}

function RibbonSVG({ color, size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" fill="none">
      <path d="M9 2 C6.5 2 4.5 4 4.5 6 C4.5 9 9 11 9 11 C9 11 13.5 9 13.5 6 C13.5 4 11.5 2 9 2Z" fill={color}/>
      <path d="M9 11 L6.5 17 L9 14 L11.5 17 Z" fill={color} opacity="0.75"/>
    </svg>
  );
}

// ─── 3D Card ─────────────────────────────────────────────────────────────────
function Card3D({ children, theme, style = {}, hover = true }) {
  const [hov, setHov] = useState(false);
  return (
    <div
      onMouseEnter={() => hover && setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: theme.cardBg,
        border: `1px solid ${theme.border}`,
        borderRadius: 14,
        boxShadow: hov
          ? `0 12px 32px ${theme.shadow}, 0 2px 8px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.9)`
          : `0 4px 16px ${theme.shadow}, 0 1px 4px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)`,
        transform: hov ? "translateY(-3px)" : "translateY(0)",
        transition: "all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// ─── Pipeline step card ───────────────────────────────────────────────────────
function PipelineStepCard({ step, theme, isActive, isDone }) {
  const color  = STEP_COLORS[step.step] || "#888";
  const status = isDone ? (step.success ? "done" : "error") : isActive ? "running" : "pending";

  return (
    <Card3D theme={theme} style={{ overflow: "hidden" }}>
      {/* Step header */}
      <div style={{
        padding: "12px 14px 10px",
        borderBottom: `1px solid ${theme.border}`,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: `linear-gradient(135deg, ${color}12, ${color}06)`,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: color, display: "flex", alignItems: "center",
            justifyContent: "center", color: "#fff", fontSize: 12, fontWeight: 800,
            boxShadow: `0 3px 8px ${color}55`,
          }}>
            {step.step}
          </div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: theme.text }}>
              {step.name}
            </div>
            <div style={{ fontSize: 9, color: theme.textDim }}>
              {step.elapsed_s ? `${step.elapsed_s}s` : "—"}
            </div>
          </div>
        </div>
        <div style={{
          padding: "3px 10px", borderRadius: 12, fontSize: 9, fontWeight: 700,
          letterSpacing: 0.8,
          background: status === "done" ? "#D1FAE5" : status === "error" ? "#FEE2E2" : status === "running" ? "#DBEAFE" : "#F1F5F9",
          color: status === "done" ? "#065F46" : status === "error" ? "#991B1B" : status === "running" ? "#1D4ED8" : "#64748B",
          border: `1px solid ${status === "done" ? "#6EE7B7" : status === "error" ? "#FECACA" : status === "running" ? "#93C5FD" : "#E2E8F0"}`,
        }}>
          {status === "done" ? "✓ DONE" : status === "error" ? "✗ ERROR" : status === "running" ? "RUNNING" : "PENDING"}
        </div>
      </div>

      {/* Description */}
      <div style={{ padding: "8px 14px 10px", fontSize: 10, color: theme.textDim, lineHeight: 1.5 }}>
        {step.description}
        {step.error && (
          <div style={{ marginTop: 4, color: "#DC2626", fontSize: 9 }}>{step.error.slice(0, 80)}</div>
        )}
      </div>

      {/* Slice images — 3 views */}
      {step.slices && (
        <div style={{ padding: "0 10px 12px", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
          {["axial", "coronal", "sagittal"].map(view => (
            <div key={view} style={{ position: "relative" }}>
              {step.slices[view] ? (
                <img
                  src={`data:image/png;base64,${step.slices[view]}`}
                  alt={view}
                  style={{
                    width: "100%", aspectRatio: "1",
                    borderRadius: 8, objectFit: "cover",
                    border: `1px solid ${theme.border}`,
                    boxShadow: `0 2px 6px rgba(0,0,0,0.12)`,
                    display: "block",
                  }}
                />
              ) : (
                <div style={{
                  width: "100%", aspectRatio: "1", borderRadius: 8,
                  background: "#0a0a0a", border: `1px solid ${theme.border}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 9, color: "#555",
                }}>
                  {view}
                </div>
              )}
              <div style={{
                position: "absolute", bottom: 4, left: 0, right: 0,
                textAlign: "center", fontSize: 8, color: "rgba(255,255,255,0.7)",
                textTransform: "uppercase", letterSpacing: 0.5,
                textShadow: "0 1px 2px rgba(0,0,0,0.8)",
              }}>
                {view}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card3D>
  );
}

// ─── Result panel ─────────────────────────────────────────────────────────────
function ResultCard({ prediction, theme }) {
  const cfg = CLASS_CONFIG[prediction.prediction] || {};
  const riskColors = {
    HIGH:     { bg: "#FEE2E2", text: "#991B1B", border: "#FECACA" },
    MODERATE: { bg: "#FEF9C3", text: "#854D0E", border: "#FDE68A" },
    LOW:      { bg: "#DCFCE7", text: "#166534", border: "#BBF7D0" },
  };
  const rc = riskColors[prediction.risk_level] || riskColors.LOW;

  return (
    <Card3D theme={theme} style={{ padding: "18px 20px" }} hover={false}>
      <div style={{ fontSize: 10, color: theme.textDim, letterSpacing: 2, textTransform: "uppercase", fontWeight: 700, marginBottom: 14 }}>
        Final Prediction
      </div>

      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 40, fontWeight: 900, color: theme.accentStrong, lineHeight: 1, marginBottom: 4 }}>
            {prediction.prediction}
          </div>
          <div style={{ fontSize: 11, color: theme.textDim }}>{prediction.confidence?.toFixed(1)}% confidence</div>
        </div>
        <div style={{
          padding: "6px 14px", borderRadius: 20,
          background: rc.bg, border: `1px solid ${rc.border}`,
          display: "flex", alignItems: "center", gap: 6,
        }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: rc.text }}/>
          <span style={{ fontSize: 10, fontWeight: 700, color: rc.text, letterSpacing: 0.8 }}>
            {prediction.risk_level} RISK
          </span>
        </div>
      </div>

      <div style={{ fontSize: 12, color: theme.textMuted, marginBottom: 14 }}>{cfg.label}</div>

      {prediction.class_scores?.map((s, i) => (
        <div key={s.label} style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 12, color: theme.textMuted }}>{s.label}</span>
            <span style={{ fontSize: 12, fontWeight: i === 0 ? 700 : 400, color: i === 0 ? theme.accentStrong : theme.textDim }}>
              {(s.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 8, background: theme.barBg, borderRadius: 4, overflow: "hidden", border: `1px solid ${theme.border}` }}>
            <div style={{
              height: "100%", width: `${s.confidence * 100}%`,
              background: i === 0 ? theme.accent : theme.accentMid,
              borderRadius: 4, opacity: i === 0 ? 1 : 0.45,
              transition: "width 1s cubic-bezier(0.16,1,0.3,1)",
            }}/>
          </div>
        </div>
      ))}

      <div style={{ marginTop: 14, padding: "10px 12px", borderRadius: 8, background: theme.accentLight, border: `1px solid ${theme.border}` }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
          <span style={{ fontSize: 11, color: theme.textMuted }}>z_img embedding</span>
          <span style={{ fontSize: 11, fontWeight: 700, color: theme.accentStrong }}>256-d L2-normalised</span>
        </div>
        <div style={{ fontSize: 10, color: theme.textDim }}>Ready for multimodal fusion engine →</div>
      </div>

      <div style={{ marginTop: 12, padding: "8px 10px", borderRadius: 6, background: "#FFFDE7", border: "1px solid #F9A825", fontSize: 10, color: "#5D4037", lineHeight: 1.5 }}>
        ⚠ Non-diagnostic output only · Not a substitute for clinical evaluation
      </div>
    </Card3D>
  );
}

// ─── Upload zone ──────────────────────────────────────────────────────────────
function UploadZone({ onFile, loading, theme, mode }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();
  const drop = useCallback((e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0]; if (f) onFile(f);
  }, [onFile]);

  const accept  = mode === "dicom" ? ".zip" : ".nii,.nii.gz";
  const label   = mode === "dicom" ? "Drop DICOM zip here" : "Drop NIfTI file here";
  const sublabel = mode === "dicom" ? "ZIP of DICOM folder · Full pipeline + prediction" : ".nii · .nii.gz · preprocessed T1 MRI";

  return (
    <div
      onClick={() => !loading && ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={drop}
      style={{
        border: `2px dashed ${drag ? theme.accent : theme.borderLight}`,
        borderRadius: 12, padding: "28px 16px", textAlign: "center",
        cursor: loading ? "not-allowed" : "pointer",
        background: drag ? theme.accentLight : theme.barBg,
        transition: "all 0.2s",
        boxShadow: drag ? `0 4px 16px ${theme.shadow}` : "none",
      }}
    >
      <input ref={ref} type="file" accept={accept} style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f); }}/>
      <div style={{ marginBottom: 10 }}><DiseaseIcon theme={theme} size={36}/></div>
      <div style={{ color: theme.text, fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
        {loading ? "Processing…" : label}
      </div>
      <div style={{ color: theme.textDim, fontSize: 11 }}>{sublabel}</div>
    </div>
  );
}

// ─── Mode toggle ──────────────────────────────────────────────────────────────
function ModeToggle({ mode, setMode, theme }) {
  return (
    <div style={{
      display: "flex", background: theme.barBg,
      border: `1px solid ${theme.border}`, borderRadius: 10,
      padding: 3, gap: 3,
    }}>
      {[
        { id: "dicom",  label: "DICOM → Pipeline",  icon: "⚡" },
        { id: "nifti",  label: "NIfTI → Predict",   icon: "🧠" },
      ].map(m => (
        <button key={m.id} onClick={() => setMode(m.id)} style={{
          flex: 1, padding: "7px 10px", borderRadius: 7, border: "none",
          background: mode === m.id ? theme.cardBg : "transparent",
          color: mode === m.id ? theme.accentStrong : theme.textDim,
          fontSize: 11, fontWeight: mode === m.id ? 700 : 400,
          cursor: "pointer", transition: "all 0.2s",
          boxShadow: mode === m.id ? `0 2px 8px ${theme.shadow}, inset 0 1px 0 rgba(255,255,255,0.9)` : "none",
        }}>
          {m.icon} {m.label}
        </button>
      ))}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [active,    setActive]    = useState("alzheimers");
  const [mode,      setMode]      = useState("dicom");
  const [file,      setFile]      = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [result,    setResult]    = useState(null);       // NIfTI mode result
  const [pipeline,  setPipeline]  = useState(null);       // DICOM mode result
  const [activeStep, setActiveStep] = useState(-1);

  const t = THEMES[active];

  const switchDisease = (id) => {
    setActive(id); setResult(null); setPipeline(null);
    setFile(null); setError(null); setActiveStep(-1);
  };

  const handleModeChange = (m) => {
    setMode(m); setResult(null); setPipeline(null);
    setFile(null); setError(null); setActiveStep(-1);
  };

  const handleFile = useCallback(async (f) => {
    setFile(f); setResult(null); setPipeline(null);
    setError(null); setLoading(true); setActiveStep(0);

    try {
      if (mode === "dicom") {
        // Simulate step progress while waiting
        const timer = setInterval(() => {
          setActiveStep(s => Math.min(s + 1, 3));
        }, 8000);
        const data = await preprocessDICOM(f);
        clearInterval(timer);
        setActiveStep(-1);
        setPipeline(data);
      } else {
        const data = await analyzeMRI(f);
        setResult(data);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setActiveStep(-1);
    }
  }, [mode]);

  const hasPipeline = pipeline && pipeline.pipeline;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=DM+Serif+Display&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${t.bg}; color: ${t.text}; font-family: 'Inter', sans-serif; min-height: 100vh; transition: background 0.3s; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${t.border}; border-radius: 3px; }
      `}</style>

      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>

        {/* Header */}
        <header style={{ background: t.headerBg, padding: "0 28px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 0 10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 40, height: 40, borderRadius: 10,
                background: "rgba(255,255,255,0.2)", border: "1px solid rgba(255,255,255,0.35)",
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: "0 4px 12px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3)",
              }}>
                <DiseaseIcon theme={t} size={24}/>
              </div>
              <div>
                <div style={{ fontSize: 17, fontWeight: 900, color: "#fff", fontFamily: "'DM Serif Display', serif", letterSpacing: -0.3 }}>
                  NeuroRisk AI
                </div>
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.7)", letterSpacing: 1.5, textTransform: "uppercase" }}>
                  R26-DS-015 · Adaptive Multimodal Neurological Risk Assessment
                </div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {Object.values(THEMES).map(d => (
                <button key={d.id} onClick={() => switchDisease(d.id)} style={{
                  padding: "7px 16px", borderRadius: 20,
                  border: active === d.id ? "1.5px solid rgba(255,255,255,0.85)" : "1.5px solid rgba(255,255,255,0.25)",
                  background: active === d.id ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.1)",
                  color: "#fff", fontSize: 12, fontWeight: active === d.id ? 700 : 400,
                  cursor: "pointer", transition: "all 0.2s",
                  display: "flex", alignItems: "center", gap: 6,
                  boxShadow: active === d.id ? "0 2px 8px rgba(0,0,0,0.2)" : "none",
                }}>
                  <RibbonSVG color={d.id === "parkinsons" ? "#E0E0E0" : "#fff"} size={13}/>
                  {d.short}
                  {d.id !== "alzheimers" && (
                    <span style={{ fontSize: 8, background: "rgba(255,255,255,0.18)", color: "rgba(255,255,255,0.7)", padding: "1px 5px", borderRadius: 4 }}>
                      SOON
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Banner */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0 12px", borderTop: "1px solid rgba(255,255,255,0.15)" }}>
            <RibbonSVG color={t.id === "parkinsons" ? "#E0E0E0" : "rgba(255,255,255,0.9)"} size={15}/>
            <span style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.95)" }}>{t.name}</span>
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>·</span>
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.65)" }}>{t.tagline}</span>
            <div style={{ display: "flex", gap: 5, marginLeft: 4 }}>
              {[...Array(6)].map((_, i) => (
                <span key={i} style={{ fontSize: 15, display: "inline-block", animation: `float ${2 + i * 0.25}s ease-in-out infinite`, animationDelay: `${i * 0.18}s` }}>
                  {t.decorSymbol}
                </span>
              ))}
            </div>
          </div>
        </header>

        {/* Body */}
        <div style={{ display: "flex", flex: 1 }}>

          {/* Left sidebar */}
          <aside style={{
            width: 290, background: t.sidebar,
            borderRight: `1px solid ${t.border}`,
            padding: 18, display: "flex", flexDirection: "column", gap: 16,
            boxShadow: `2px 0 12px ${t.shadow}`,
          }}>
            {/* Mode toggle */}
            <div>
              <div style={{ fontSize: 9, color: t.textDim, letterSpacing: 2, textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
                Input Mode
              </div>
              <ModeToggle mode={mode} setMode={handleModeChange} theme={t}/>
            </div>

            {/* Upload */}
            <div>
              <div style={{ fontSize: 9, color: t.textDim, letterSpacing: 2, textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
                {mode === "dicom" ? "Upload · DICOM Zip" : "Upload · NIfTI"}
              </div>
              <UploadZone onFile={handleFile} loading={loading} theme={t} mode={mode}/>
            </div>

            {file && (
              <Card3D theme={t} style={{ padding: "10px 12px" }} hover={false}>
                <div style={{ fontSize: 9, color: t.textDim, marginBottom: 4, letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>Selected</div>
                <div style={{ fontSize: 11, color: t.text, fontWeight: 600, wordBreak: "break-all" }}>{file.name}</div>
                <div style={{ fontSize: 10, color: t.textDim, marginTop: 3 }}>{(file.size / 1048576).toFixed(1)} MB</div>
              </Card3D>
            )}

            {/* Pipeline progress */}
            {loading && mode === "dicom" && (
              <div>
                <div style={{ fontSize: 9, color: t.textDim, letterSpacing: 2, textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
                  Pipeline Progress
                </div>
                {STEP_NAMES.map((name, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "6px 0", borderBottom: `1px solid ${t.border}`,
                    fontSize: 11,
                  }}>
                    <div style={{
                      width: 20, height: 20, borderRadius: 6,
                      background: i < activeStep ? "#D1FAE5" : i === activeStep ? STEP_COLORS[i] : t.barBg,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 9, fontWeight: 800, color: i < activeStep ? "#065F46" : i === activeStep ? "#fff" : t.textDim,
                      animation: i === activeStep ? "pulse 1s ease infinite" : "none",
                    }}>
                      {i < activeStep ? "✓" : i}
                    </div>
                    <span style={{ color: i <= activeStep ? t.text : t.textDim, fontWeight: i === activeStep ? 600 : 400 }}>
                      {name}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Disease selector */}
            <div>
              <div style={{ fontSize: 9, color: t.textDim, letterSpacing: 2, textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
                Analysis Target
              </div>
              {Object.values(THEMES).map(d => (
                <div key={d.id} onClick={() => switchDisease(d.id)} style={{
                  padding: "9px 12px", marginBottom: 4, borderRadius: 8,
                  background: active === d.id ? d.accentLight : "transparent",
                  border: `1px solid ${active === d.id ? d.borderLight : "transparent"}`,
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  cursor: "pointer", transition: "all 0.2s",
                  boxShadow: active === d.id ? `0 2px 8px ${d.shadow}` : "none",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <RibbonSVG color={d.ribbon} size={14}/>
                    <span style={{ fontSize: 12, color: active === d.id ? d.accentStrong : t.textDim, fontWeight: active === d.id ? 700 : 400 }}>
                      {d.short}
                    </span>
                  </div>
                  <span style={{ fontSize: 9, color: active === d.id ? d.accent : t.textDim, fontWeight: 700, letterSpacing: 1 }}>
                    {active === d.id ? "ACTIVE" : "SOON"}
                  </span>
                </div>
              ))}
            </div>
          </aside>

          {/* Main content */}
          <main style={{ flex: 1, padding: 24, background: t.bg, overflowY: "auto" }}>

            {/* Empty state */}
            {!loading && !hasPipeline && !result && !error && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", minHeight: 400 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ animation: "float 3s ease-in-out infinite", marginBottom: 20 }}>
                    <DiseaseIcon theme={t} size={80}/>
                  </div>
                  <div style={{ fontSize: 18, color: t.text, fontWeight: 700, marginBottom: 8 }}>
                    {mode === "dicom" ? "Upload a DICOM zip to begin" : "Upload a NIfTI file to begin"}
                  </div>
                  <div style={{ fontSize: 12, color: t.textMuted, marginBottom: 4 }}>{t.name} · Brain MRI Analysis</div>
                  <div style={{ fontSize: 11, color: t.textDim }}>{t.tagline}</div>
                  {mode === "dicom" && (
                    <div style={{ marginTop: 16, padding: "10px 20px", borderRadius: 8, background: t.accentLight, border: `1px solid ${t.border}`, display: "inline-block" }}>
                      <div style={{ fontSize: 11, color: t.textMuted }}>
                        Full pipeline: DICOM → NIfTI → Registration → Skull Strip → Bias Correct → Prediction
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", minHeight: 400 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{
                    width: 56, height: 56, margin: "0 auto 20px",
                    border: `4px solid ${t.accentLight}`,
                    borderTop: `4px solid ${t.accent}`,
                    borderRadius: "50%",
                    animation: "spin 0.9s linear infinite",
                    boxShadow: `0 0 20px ${t.shadow}`,
                  }}/>
                  <div style={{ fontSize: 15, color: t.text, fontWeight: 700, marginBottom: 6 }}>
                    {mode === "dicom" ? `Running Step ${activeStep}: ${STEP_NAMES[activeStep] || "Processing"}…` : "Analyzing MRI…"}
                  </div>
                  <div style={{ fontSize: 11, color: t.textDim }}>
                    {mode === "dicom" ? "FSL + ANTs preprocessing pipeline" : "VisionEncoder · 3D ResNet-18 · 256-d z_img"}
                  </div>
                  {mode === "dicom" && (
                    <div style={{ marginTop: 12, fontSize: 10, color: t.textDim }}>
                      This may take 10–30 minutes depending on scan size
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <Card3D theme={t} style={{ padding: "20px 24px", maxWidth: 600, margin: "40px auto" }} hover={false}>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#991B1B", marginBottom: 8 }}>
                  ✕ Processing Error
                </div>
                <div style={{ fontSize: 12, color: "#7F1D1D", lineHeight: 1.6 }}>{error}</div>
                <button
                  onClick={() => { setError(null); setFile(null); }}
                  style={{ marginTop: 14, padding: "7px 16px", borderRadius: 8, border: "none", background: t.accent, color: "#fff", fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                >
                  Try Again
                </button>
              </Card3D>
            )}

            {/* NIfTI result */}
            {result && !hasPipeline && (
              <div style={{ maxWidth: 520, margin: "0 auto", animation: "fadeUp 0.5s ease both" }}>
                <ResultCard prediction={result} theme={t}/>
              </div>
            )}

            {/* Pipeline result */}
            {hasPipeline && (
              <div style={{ animation: "fadeUp 0.5s ease both" }}>

                {/* Pipeline header */}
                <div style={{ marginBottom: 20, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div>
                    <div style={{ fontSize: 16, fontWeight: 800, color: t.text, marginBottom: 2 }}>
                      Preprocessing Pipeline
                    </div>
                    <div style={{ fontSize: 11, color: t.textDim }}>
                      {pipeline.subject_id} · {pipeline.total_elapsed_s}s total
                      {!pipeline.pipeline_success && <span style={{ color: "#DC2626", marginLeft: 8 }}>— Pipeline error</span>}
                    </div>
                  </div>
                  <div style={{
                    padding: "5px 14px", borderRadius: 20, fontSize: 11, fontWeight: 700,
                    background: pipeline.pipeline_success ? "#D1FAE5" : "#FEE2E2",
                    color: pipeline.pipeline_success ? "#065F46" : "#991B1B",
                    border: `1px solid ${pipeline.pipeline_success ? "#6EE7B7" : "#FECACA"}`,
                  }}>
                    {pipeline.pipeline_success ? "✓ Pipeline Complete" : "✗ Pipeline Failed"}
                  </div>
                </div>

                {/* Step grid — 2 columns */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                  {pipeline.pipeline.map((step) => (
                    <PipelineStepCard
                      key={step.step}
                      step={step}
                      theme={t}
                      isActive={false}
                      isDone={true}
                    />
                  ))}
                </div>

                {/* Prediction */}
                {pipeline.prediction && !pipeline.prediction.error && (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: t.textMuted, letterSpacing: 1, textTransform: "uppercase", marginBottom: 12 }}>
                      → Inference Result
                    </div>
                    <ResultCard prediction={pipeline.prediction} theme={t}/>
                  </div>
                )}

                {pipeline.prediction?.error && (
                  <Card3D theme={t} style={{ padding: "14px 18px" }} hover={false}>
                    <div style={{ fontSize: 12, color: "#DC2626" }}>
                      Inference failed: {pipeline.prediction.error}
                    </div>
                  </Card3D>
                )}
              </div>
            )}
          </main>
        </div>

        {/* Footer */}
        <footer style={{
          padding: "10px 28px", background: t.sidebar,
          borderTop: `1px solid ${t.border}`,
          display: "flex", justifyContent: "space-between",
          fontSize: 10, color: t.textDim,
          boxShadow: `0 -2px 8px ${t.shadow}`,
        }}>
          <span>R26-DS-015 | Dissanayaka D.M.T.M | IT22203762 </span>
          <span>NON-DIAGNOSTIC DECISION-SUPPORT OUTPUT ONLY</span>
        </footer>
      </div>
    </>
  );
}
