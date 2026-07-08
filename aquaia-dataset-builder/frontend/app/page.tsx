"use client";

import { useEffect } from "react";
import { useAppStore } from "@/store/appStore";
import Sidebar from "@/components/sidebar/Sidebar";
import WorkspaceSelector from "@/components/workspace/WorkspaceSelector";
import WorkspaceLoginModal from "@/components/workspace/WorkspaceLoginModal";
import DashboardPanel from "@/features/dashboard/DashboardPanel";
import SearchPanel from "@/features/search/SearchPanel";
import ValidationPanel from "@/features/validation/ValidationPanel";
import DatasetPanel from "@/features/dataset/DatasetPanel";
import ExportPanel from "@/features/export/ExportPanel";
import SettingsPanel from "@/features/settings/SettingsPanel";

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
  const { initWorkspace, workspaceReady } = useAppStore();

  useEffect(() => {
    initWorkspace();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg-base)]">
      <WorkspaceLoginModal />
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header
          className="shrink-0 flex items-center justify-between px-6 py-3 border-b"
          style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold" style={{ color: "var(--text-base)" }}>
              ADIAB
            </span>
            <span className="text-sm" style={{ color: "var(--text-dim)" }}>
              AquaIA Dataset Builder
            </span>
            <span
              className="ml-1 px-1.5 py-0.5 text-[10px] font-mono rounded border"
              style={{ color: "var(--text-muted)", borderColor: "var(--border)", background: "var(--bg-input)" }}
            >
              v0.2.0
            </span>
          </div>
          <WorkspaceSelector />
        </header>

        <main className="flex-1 overflow-y-auto p-6">
          {workspaceReady ? (
            <ActivePanel />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
