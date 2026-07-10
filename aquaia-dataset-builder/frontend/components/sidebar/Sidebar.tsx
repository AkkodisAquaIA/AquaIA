"use client";

import {
  LayoutDashboard, Search, CheckSquare, Database,
  Download, Settings, Microscope, Sun, Moon, BookOpen,
} from "lucide-react";
import { useAppStore } from "@/store/appStore";
import type { PanelId } from "@/types";
import { cn } from "@/lib/utils";

const NAV_ITEMS: { id: PanelId; label: string; icon: React.ElementType }[] = [
  { id: "dashboard",  label: "Dashboard",        icon: LayoutDashboard },
  { id: "search",     label: "Search",            icon: Search },
  { id: "validation", label: "Validation Queue",  icon: CheckSquare },
  { id: "dataset",    label: "Dataset Explorer",  icon: Database },
  { id: "export",     label: "Export Center",     icon: Download },
  { id: "settings",   label: "Settings",          icon: Settings },
  { id: "docs",       label: "Documentation",     icon: BookOpen },
];

const READONLY_HIDDEN: PanelId[] = ["search", "validation", "settings"];
const ALWAYS_VISIBLE: PanelId[] = ["docs"];

export default function Sidebar() {
  const { activePanel, setActivePanel, theme, toggleTheme, isReadOnly } = useAppStore();
  const visibleItems = isReadOnly
    ? NAV_ITEMS.filter((item) => !READONLY_HIDDEN.includes(item.id) || ALWAYS_VISIBLE.includes(item.id))
    : NAV_ITEMS;

  return (
    <aside
      className="flex flex-col w-56 h-screen shrink-0 border-r transition-colors duration-200"
      style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-green-500/10 border border-green-500/30">
          <Microscope className="w-4 h-4 text-green-400" />
        </div>
        <div>
          <p className="text-sm font-semibold" style={{ color: "var(--text-base)" }}>ADIAB</p>
          <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>AquaIA Dataset Builder</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {visibleItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActivePanel(id)}
            className={cn(
              "flex items-center gap-3 w-full px-3 py-2 rounded-lg text-sm transition-all duration-150",
              activePanel === id
                ? "bg-green-500/10 text-green-400 border border-green-500/20"
                : "border border-transparent"
            )}
            style={activePanel !== id ? { color: "var(--text-dim)" } : undefined}
            onMouseEnter={(e) => {
              if (activePanel !== id)
                (e.currentTarget as HTMLElement).style.background = "var(--bg-input)";
            }}
            onMouseLeave={(e) => {
              if (activePanel !== id)
                (e.currentTarget as HTMLElement).style.background = "";
            }}
          >
            <Icon className="w-4 h-4 shrink-0" />
            <span className="font-medium">{label}</span>
          </button>
        ))}
      </nav>

      {/* Read-only notice */}
      {isReadOnly && (
        <div className="mx-2 mb-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
          <p className="text-[10px] text-amber-400 font-medium">Read-only mode</p>
          <p className="text-[9px] text-amber-400/70 mt-0.5">Login to edit this workspace</p>
        </div>
      )}

      {/* Footer — theme toggle */}
      <div
        className="px-3 py-3 border-t flex items-center justify-between"
        style={{ borderColor: "var(--border)" }}
      >
        <p className="text-[10px]" style={{ color: "var(--text-ghost)" }}>AquaIA · v0.1.0</p>
        <button
          onClick={toggleTheme}
          className="p-1.5 rounded-lg border transition-colors"
          style={{
            background: "var(--bg-input)",
            borderColor: "var(--border)",
            color: "var(--text-dim)",
          }}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>
      </div>
    </aside>
  );
}
