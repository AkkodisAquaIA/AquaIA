"use client";

import { useEffect, useState } from "react";
import { Images, CheckCircle, XCircle, Copy, Layers, Clock } from "lucide-react";
import { getStats } from "@/lib/api";
import type { DashboardStats } from "@/types";
import { formatNumber, formatDate } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: number;
  icon: React.ElementType;
  color: string;
}

function StatCard({ label, value, icon: Icon, color }: StatCardProps) {
  return (
    <div className="bg-[#111118] border border-[#2a2a3a] rounded-xl p-4 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-2xl font-bold text-white">{formatNumber(value)}</p>
        <p className="text-xs text-[#888899] mt-0.5">{label}</p>
      </div>
    </div>
  );
}

export default function DashboardPanel() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats()
      .then(setStats)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-64 text-[#888899]">
        Failed to load stats. Is the backend running?
      </div>
    );
  }

  const total = stats.total_images;
  const validatedPct = total > 0 ? Math.round((stats.validated / total) * 100) : 0;

  return (
    <div className="panel-enter space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Dashboard</h1>
        <p className="text-sm text-[#888899] mt-1">Overview of your dataset pipeline</p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        <StatCard label="Total images" value={stats.total_images} icon={Images} color="bg-blue-500/10 text-blue-400" />
        <StatCard label="Validated" value={stats.validated} icon={CheckCircle} color="bg-green-500/10 text-green-400" />
        <StatCard label="Rejected" value={stats.rejected} icon={XCircle} color="bg-red-500/10 text-red-400" />
        <StatCard label="Duplicates" value={stats.duplicates} icon={Copy} color="bg-purple-500/10 text-purple-400" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Taxons" value={stats.total_taxons} icon={Layers} color="bg-yellow-500/10 text-yellow-400" />
        <StatCard label="Exports" value={stats.total_exports} icon={Clock} color="bg-orange-500/10 text-orange-400" />
      </div>

      {/* Progress bar */}
      {total > 0 && (
        <div className="bg-[#111118] border border-[#2a2a3a] rounded-xl p-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-[#888899]">Validation progress</span>
            <span className="text-white font-medium">{validatedPct}%</span>
          </div>
          <div className="w-full bg-[#2a2a3a] rounded-full h-2">
            <div
              className="bg-green-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${validatedPct}%` }}
            />
          </div>
          <p className="text-xs text-[#888899] mt-2">
            {formatNumber(stats.validated)} validated · {formatNumber(stats.pending)} pending
          </p>
        </div>
      )}

      {/* Recent searches */}
      {stats.recent_searches.length > 0 && (
        <div className="bg-[#111118] border border-[#2a2a3a] rounded-xl p-4">
          <h2 className="text-sm font-semibold text-white mb-3">Recent searches</h2>
          <div className="space-y-2">
            {stats.recent_searches.map((s) => (
              <div key={s.id} className="flex items-center justify-between text-sm">
                <span className="text-[#ccccdd] font-mono">{s.query}</span>
                <div className="flex items-center gap-3">
                  <span className="text-[#888899]">{s.result_count} results</span>
                  <span className="text-[#555566] text-xs">{formatDate(s.created_at)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {total === 0 && (
        <div className="bg-[#111118] border border-[#2a2a3a] rounded-xl p-8 text-center">
          <Images className="w-12 h-12 text-[#333344] mx-auto mb-3" />
          <p className="text-[#888899] text-sm">No images yet.</p>
          <p className="text-[#555566] text-xs mt-1">Use the Search panel to retrieve images.</p>
        </div>
      )}
    </div>
  );
}
