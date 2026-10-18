import { BrainCircuit } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Shared platform header (blue band) — used by AppShell for every page and by
 * the neuroimaging dashboard, so the whole app has one header design.
 *
 *  leading : node before the logo (e.g. mobile menu button)
 *  actions : node on the right (status pill, avatar, disease tabs, ...)
 *  subRow  : optional thin second row under a divider
 */
export function AppHeader({
  leading,
  actions,
  subRow,
  className,
}: {
  leading?: React.ReactNode;
  actions?: React.ReactNode;
  subRow?: React.ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("bg-[#2F5FDC] text-white shadow-[0_2px_12px_rgba(16,24,40,0.12)]", className)}>
      <div className="flex min-h-[72px] items-center justify-between gap-4 px-4 sm:px-7">
        <div className="flex min-w-0 items-center gap-3">
          {leading}
          <div className="grid size-10 shrink-0 place-items-center rounded-[10px] border border-white/35 bg-white/20 shadow-[0_4px_12px_rgba(0,0,0,0.15)]">
            <BrainCircuit className="size-6" />
          </div>
          <div className="min-w-0">
            <div className="text-[17px] font-bold leading-tight tracking-tight">NeuroRisk AI</div>
            <div className="hidden text-[9px] font-medium uppercase tracking-[0.15em] text-white/70 md:block">
              R26-DS-015 · Adaptive Multimodal Neurological Risk Assessment
            </div>
          </div>
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-3 sm:gap-5">{actions}</div> : null}
      </div>
      {subRow ? <div className="border-t border-white/15 px-4 py-2 sm:px-7">{subRow}</div> : null}
    </header>
  );
}