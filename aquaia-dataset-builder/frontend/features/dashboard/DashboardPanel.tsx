"use client";

import { useEffect, useState } from "react";
import { Images, CheckCircle, XCircle, Copy, Layers, Clock, HardDrive } from "lucide-react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
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
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-2xl font-bold text-[var(--text-base)]">{formatNumber(value)}</p>
        <p className="text-xs text-[var(--text-dim)] mt-0.5">{label}</p>
      </div>
    </div>
  );
}

const STATUS_COLORS: Record<string, string> = {
  Validated: "#22c55e",
  Pending: "#f59e0b",
  Rejected: "#ef4444",
  Duplicates: "#a855f7",
};

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
      <div className="flex items-center justify-center h-64 text-[var(--text-dim)]">
        Failed to load stats. Is the backend running?
      </div>
    );
  }

  const total = stats.total_images;
  const validatedPct = total > 0 ? Math.round((stats.validated / total) * 100) : 0;

  const pieData = [
    { name: "Validated", value: stats.validated },
    { name: "Pending", value: stats.pending },
    { name: "Rejected", value: stats.rejected },
    { name: "Duplicates", value: stats.duplicates },
  ].filter((d) => d.value > 0);

  return (
    <div className="panel-enter space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Dashboard</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Overview of your dataset pipeline</p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        <StatCard label="Total images" value={stats.total_images} icon={Images} color="bg-blue-500/10 text-blue-400" />
        <StatCard label="Validated" value={stats.validated} icon={CheckCircle} color="bg-green-500/10 text-green-400" />
        <StatCard label="Rejected" value={stats.rejected} icon={XCircle} color="bg-red-500/10 text-red-400" />
        <StatCard label="Duplicates" value={stats.duplicates} icon={Copy} color="bg-purple-500/10 text-purple-400" />
      </div>

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        <StatCard label="Taxons" value={stats.total_taxons} icon={Layers} color="bg-yellow-500/10 text-yellow-400" />
        <StatCard label="Exports" value={stats.total_exports} icon={Clock} color="bg-orange-500/10 text-orange-400" />
        <StatCard label="Downloaded" value={stats.downloaded ?? 0} icon={HardDrive} color="bg-cyan-500/10 text-cyan-400" />
      </div>

      {/* Chart + Progress */}
      {total > 0 && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {/* Pie chart */}
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4">
            <p className="text-sm font-semibold text-[var(--text-base)] mb-3">Status distribution</p>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {pieData.map((entry) => (
                    <Cell key={entry.name} fill={STATUS_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                    color: "var(--text-base)",
                    fontSize: "12px",
                  }}
                />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{ fontSize: "11px", color: "var(--text-dim)" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Validation progress */}
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 flex flex-col justify-center gap-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-[var(--text-dim)]">Validation progress</span>
                <span className="text-[var(--text-base)] font-medium">{validatedPct}%</span>
              </div>
              <div className="w-full bg-[var(--border)] rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${validatedPct}%` }}
                />
              </div>
              <p className="text-xs text-[var(--text-dim)] mt-2">
                {formatNumber(stats.validated)} validated · {formatNumber(stats.pending)} pending
              </p>
            </div>

            {stats.downloaded !== undefined && total > 0 && (
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-[var(--text-dim)]">Downloaded locally</span>
                  <span className="text-[var(--text-base)] font-medium">
                    {Math.round(((stats.downloaded ?? 0) / total) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-[var(--border)] rounded-full h-2">
                  <div
                    className="bg-cyan-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(((stats.downloaded ?? 0) / total) * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-[var(--text-dim)] mt-2">
                  {formatNumber(stats.downloaded ?? 0)} / {formatNumber(total)} images cached on disk
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recent searches */}
      {stats.recent_searches.length > 0 && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4">
          <h2 className="text-sm font-semibold text-[var(--text-base)] mb-3">Recent searches</h2>
          <div className="space-y-2">
            {stats.recent_searches.map((s) => (
              <div key={s.id} className="flex items-center justify-between text-sm">
                <span className="text-[var(--text-sub)] font-mono">{s.query}</span>
                <div className="flex items-center gap-3">
                  <span className="text-[var(--text-dim)]">{s.result_count} results</span>
                  <span className="text-[var(--text-muted)] text-xs">{formatDate(s.created_at)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {total === 0 && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-8 text-center">
          <Images className="w-12 h-12 text-[var(--text-ghost)] mx-auto mb-3" />
          <p className="text-[var(--text-dim)] text-sm">No images yet.</p>
          <p className="text-[var(--text-muted)] text-xs mt-1">Use the Search panel to retrieve images.</p>
        </div>
      )}
    </div>
  );
}
