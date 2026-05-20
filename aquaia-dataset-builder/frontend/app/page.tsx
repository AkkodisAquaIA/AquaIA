"use client";

import { useAppStore } from "@/store/appStore";
import Sidebar from "@/components/sidebar/Sidebar";
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
  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0a0f]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <ActivePanel />
      </main>
    </div>
  );
}
