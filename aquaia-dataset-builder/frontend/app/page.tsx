"use client";

import { useAppStore } from "@/store/appStore";
import Sidebar from "@/components/sidebar/Sidebar";
import DashboardPanel from "@/features/dashboard/DashboardPanel";
import SearchPanel from "@/features/search/SearchPanel";
import ValidationPanel from "@/features/validation/ValidationPanel";
import DatasetPanel from "@/features/dataset/DatasetPanel";
import ExportPanel from "@/features/export/ExportPanel";
import SettingsPanel from "@/features/settings/SettingsPanel";
import { Microscope } from "lucide-react";

function ActivePanel() {
  const { activePanel } = useAppStore();
  switch (activePanel) {
    case "dashboard":   return <DashboardPanel />;
    case "search":      return <SearchPanel />;
    case "validation":  return <ValidationPanel />;
    case "dataset":     return <DatasetPanel />;
    case "export":      return <ExportPanel />;
    case "settings":    return <SettingsPanel />;
  }
}

export default function Home() {
  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg-base)]">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="shrink-0 flex items-center justify-end px-6 py-3 border-b"
          style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <div className="flex items-center gap-2">
            <div className="flex items-center justify-center w-6 h-6 rounded-md bg-green-500/10 border border-green-500/30">
              <Microscope className="w-3.5 h-3.5 text-green-400" />
            </div>
            <span className="text-sm font-semibold" style={{ color: "var(--text-base)" }}>
              AquaIA
            </span>
            <span className="text-sm" style={{ color: "var(--text-dim)" }}>
              Dataset Builder
            </span>
            <span className="ml-1 px-1.5 py-0.5 text-[10px] font-mono rounded border"
              style={{ color: "var(--text-muted)", borderColor: "var(--border)", background: "var(--bg-input)" }}>
              v0.1.0
            </span>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          <ActivePanel />
        </main>
      </div>
    </div>
  );
}
