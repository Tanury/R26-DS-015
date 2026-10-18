"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Bell,
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
import { AppHeader } from "@/components/app-header";

const navigation = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/voice", label: "Voice Assessment", icon: Mic },
  { href: "/general", label: "Biomedical Assessment", icon: ClipboardList },
  { href: "/eeg", label: "EEG Assessment", icon: Activity },
  { href: "/neuroimaging", label: "Neuroimaging Assessment", icon: ScanSearch },
  { href: "/history", label: "Assessment History", icon: History },
  { href: "/fusion", label: "Fusion Result", icon: ChartNoAxesCombined },
];

function SideNavigation({ pathname, onNavigate }: { pathname: string; onNavigate?: () => void }) {
  return (
    <>
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
                  ? "border-blue-700 bg-blue-50 text-blue-700"
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
    <div className="min-h-screen bg-[#F5F7FB]">
      {/* Shared blue header, full width across the top */}
      <AppHeader
        className="fixed inset-x-0 top-0 z-40"
        leading={
          <Button
            variant="ghost"
            size="icon"
            className="text-white hover:bg-white/15 hover:text-white lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="size-5" />
          </Button>
        }
        actions={
          <>
            <span className="hidden items-center gap-2 rounded-full bg-white/20 px-3 py-1.5 text-xs font-semibold text-white sm:flex">
              <span className="size-2 rounded-full bg-emerald-300" /> Models Online
            </span>
            <Button variant="ghost" size="icon" className="text-white hover:bg-white/15 hover:text-white" aria-label="Notifications">
              <Bell className="size-5" />
            </Button>
            <div className="grid size-9 place-items-center rounded-full border border-white/35 bg-white/20 text-xs font-bold text-white" aria-label="Researcher profile">SC</div>
          </>
        }
      />

      {/* Sidebar sits below the header */}
      <aside className="fixed bottom-0 left-0 top-[72px] z-30 hidden w-[260px] flex-col border-r border-slate-200 bg-white lg:flex">
        <SideNavigation pathname={pathname} />
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button className="absolute inset-0 bg-slate-950/35" aria-label="Close menu" onClick={() => setMobileOpen(false)} />
          <aside className="relative flex h-full w-[290px] flex-col bg-white pt-10 shadow-xl">
            <button className="absolute right-3 top-3 rounded-md p-2 text-slate-500" onClick={() => setMobileOpen(false)} aria-label="Close navigation">
              <X className="size-5" />
            </button>
            <SideNavigation pathname={pathname} onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      <div className="pt-[72px] lg:pl-[260px]">
        <main className="mx-auto w-full max-w-[1400px] p-4 sm:p-6 lg:p-8">{children}</main>
      </div>
    </div>
  );
}