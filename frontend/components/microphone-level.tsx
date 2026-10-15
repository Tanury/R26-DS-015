"use client";

import { useEffect, useState } from "react";

export function MicrophoneLevel({ stream }: { stream: MediaStream }) {
  const [level, setLevel] = useState<number | null>(null);

  useEffect(() => {
    let context: AudioContext | undefined;
    let frame = 0;
    let disposed = false;
    async function monitor() {
      try {
        context = new AudioContext();
        await context.resume();
        if (disposed) return;
        const analyser = context.createAnalyser();
        analyser.fftSize = 1024;
        context.createMediaStreamSource(stream).connect(analyser);
        const samples = new Float32Array(analyser.fftSize);
        function update() {
          if (disposed) return;
          analyser.getFloatTimeDomainData(samples);
          const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
          // Logarithmic display of measured input; this does not alter the recording.
          setLevel(Math.round(Math.max(0, Math.min(100, (20 * Math.log10(rms || 1e-8) + 60) / 60 * 100))));
          frame = requestAnimationFrame(update);
        }
        update();
      } catch {
        // Recording still works when the browser cannot create a level meter.
        if (!disposed) setLevel(null);
      }
    }
    void monitor();
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      if (context && context.state !== "closed") void context.close().catch(() => {});
    };
  }, [stream]);

  return <div className="mt-5 w-full max-w-sm text-sm">
    <div className="mb-2 font-semibold">Microphone level</div>
    <div className="h-3 overflow-hidden rounded-full bg-slate-200" role="meter" aria-label="Microphone input level" aria-valuemin={0} aria-valuemax={100} aria-valuenow={level ?? 0}>
      <div className="h-full bg-blue-600 transition-[width]" style={{ width: `${level ?? 0}%` }} />
    </div>
    <p className="mt-2 text-xs leading-5 text-slate-600">{level === null ? "Level meter unavailable. Listen to the preview after recording." : "This bar should move when you speak. If it stays empty, check that the correct microphone is selected and unmuted."}</p>
  </div>;
}
