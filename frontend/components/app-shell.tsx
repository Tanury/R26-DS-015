"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Bell,
  BrainCircuit,
  ChartNoAxesCombined,
  CircleHelp,
  Activity,
  ClipboardList,
  History,
  LayoutDashboard,
  Menu,
  Mic,
  ScanSearch,
  Settings,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const navigation = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/voice", label: "Voice Assessment", icon: Mic },
  { href: "/general", label: "Biomedical Assessment", icon: ClipboardList },
  { href: "/eeg", label: "EEG Assessment", icon: Activity },
  {href: "/neuroimaging", label: "Neuroimaging Assessment", icon: ScanSearch },
  { href: "/history", label: "Assessment History", icon: History },
  { href: "/fusion", label: "Fusion Result", icon: ChartNoAxesCombined },
];

function SideNavigation({ pathname, onNavigate }: { pathname: string; onNavigate?: () => void }) {
  return (
    <>
      <div className="flex items-center gap-3 px-6 py-7">
        <div className="grid size-10 shrink-0 place-items-center rounded-md bg-blue-700 text-white">
          <BrainCircuit className="size-6" />
        </div>
        <div className="min-w-0">
          <div className="text-lg font-bold leading-tight text-blue-800">NeuroRisk Research Platform</div>
          <div className="text-xs font-semibold text-slate-500">Multimodal Decision Support</div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 py-4" aria-label="Primary navigation">
        {navigation.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onNavigate}
              className={cn(
                "flex h-12 items-center gap-3 border-l-4 px-6 text-sm font-medium transition-colors",
                active
                  ? "border-blue-700 bg-slate-100 text-blue-700"
                  : "border-transparent text-slate-600 hover:bg-slate-50 hover:text-slate-950",
              )}
            >
              <Icon className="size-5" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="space-y-2 border-t border-slate-200 p-5">
        <Button className="w-full" asChild>
          <Link href="/fusion"><ChartNoAxesCombined className="size-4" />Fusion Result</Link>
        </Button>
        <Link href="/support" className="flex items-center gap-3 px-2 py-2 text-sm text-slate-600 hover:text-blue-700">
          <CircleHelp className="size-5" /> Help
        </Link>
        <Link href="/settings" className="flex items-center gap-3 px-2 py-2 text-sm text-slate-600 hover:text-blue-700">
          <Settings className="size-5" /> Settings
        </Link>
      </div>
    </>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-50">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[260px] flex-col border-r border-slate-200 bg-white lg:flex">
        <SideNavigation pathname={pathname} />
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button className="absolute inset-0 bg-slate-950/35" aria-label="Close menu" onClick={() => setMobileOpen(false)} />
          <aside className="relative flex h-full w-[290px] flex-col bg-white shadow-xl">
            <button className="absolute right-3 top-3 rounded-md p-2 text-slate-500" onClick={() => setMobileOpen(false)} aria-label="Close navigation">
              <X className="size-5" />
            </button>
            <SideNavigation pathname={pathname} onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      <div className="lg:pl-[260px]">
        <header className="sticky top-0 z-20 flex h-[72px] items-center justify-between border-b border-slate-200 bg-white/95 px-4 backdrop-blur sm:px-8">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
              <Menu className="size-5" />
            </Button>
            <span className="text-lg font-semibold sm:text-xl">Research System</span>
          </div>
          <div className="flex items-center gap-3 sm:gap-5">
            <span className="hidden items-center gap-2 rounded-full bg-teal-100 px-3 py-1.5 text-xs font-semibold text-teal-800 sm:flex">
              <span className="size-2 rounded-full bg-teal-700" /> Models Online
            </span>
            <Button variant="ghost" size="icon" aria-label="Notifications"><Bell className="size-5" /></Button>
            <div className="grid size-9 place-items-center rounded-full border border-slate-300 bg-slate-100 text-xs font-bold text-slate-700" aria-label="Researcher profile">SC</div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1400px] p-4 sm:p-6 lg:p-8">{children}</main>
      </div>
    </div>
  );
}
