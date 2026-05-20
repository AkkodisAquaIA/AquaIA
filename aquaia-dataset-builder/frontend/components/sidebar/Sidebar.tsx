"use client";

import {
  LayoutDashboard,
  Search,
  CheckSquare,
  Database,
  Download,
  Settings,
  Microscope,
} from "lucide-react";
import { useAppStore } from "@/store/appStore";
import type { PanelId } from "@/types";
import { cn } from "@/lib/utils";

const NAV_ITEMS: { id: PanelId; label: string; icon: React.ElementType }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "search", label: "Search", icon: Search },
  { id: "validation", label: "Validation Queue", icon: CheckSquare },
  { id: "dataset", label: "Dataset Explorer", icon: Database },
  { id: "export", label: "Export Center", icon: Download },
  { id: "settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  const { activePanel, setActivePanel } = useAppStore();

  return (
    <aside className="flex flex-col w-56 h-screen bg-[#111118] border-r border-[#2a2a3a] shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5 border-b border-[#2a2a3a]">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-green-500/10 border border-green-500/30">
          <Microscope className="w-4 h-4 text-green-400" />
        </div>
        <div>
          <p className="text-sm font-semibold text-white tracking-tight">ADIAB</p>
          <p className="text-[10px] text-[#888899] leading-tight">Dataset Builder</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActivePanel(id)}
            className={cn(
              "flex items-center gap-3 w-full px-3 py-2 rounded-lg text-sm transition-all duration-150",
              activePanel === id
                ? "bg-green-500/10 text-green-400 border border-green-500/20"
                : "text-[#888899] hover:bg-[#1a1a24] hover:text-[#ccccdd]"
            )}
          >
            <Icon className="w-4 h-4 shrink-0" />
            <span className="font-medium">{label}</span>
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-[#2a2a3a]">
        <p className="text-[10px] text-[#555566]">AquaIA · v0.1.0</p>
      </div>
    </aside>
  );
}
