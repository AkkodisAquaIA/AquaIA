"use client";

import { useEffect, useState } from "react";
import { Download, Plus, Loader2, CheckCircle, AlertCircle, RefreshCw } from "lucide-react";
import { getExports, createExport } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ExportJob } from "@/types";
import { formatDate } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

const EXPORT_TYPES = [
  { id: "classification", label: "Classification folders", desc: "class/image.jpg structure" },
  { id: "yolo", label: "YOLO", desc: "images/ + data.yaml" },
  { id: "coco", label: "COCO JSON", desc: "instances_validated.json" },
  { id: "csv", label: "CSV", desc: "metadata spreadsheet" },
];

export default function ExportPanel() {
  const { currentUserId } = useAppStore();
  const [jobs, setJobs] = useState<ExportJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [selected, setSelected] = useState("classification");

  const reload = () => {
    if (!currentUserId) return;
    getExports(currentUserId).then(setJobs).finally(() => setLoading(false));
  };

  useEffect(() => { reload(); }, [currentUserId]);

  // Poll running jobs every 3s
  useEffect(() => {
    const running = jobs.some((j) => j.status === "running");
    if (!running || !currentUserId) return;
    const id = setInterval(() => getExports(currentUserId).then(setJobs), 3000);
    return () => clearInterval(id);
  }, [jobs, currentUserId]);

  const handleCreate = async () => {
    if (!currentUserId) return;
    setCreating(true);
    try {
      const job = await createExport(currentUserId, selected);
      setJobs((prev) => [job, ...prev]);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Export Center</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Export validated images as AI-ready datasets</p>
      </div>

      {/* Export format selector */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
        <p className="text-sm font-medium text-[var(--text-base)]">Export format</p>
        <div className="grid grid-cols-2 gap-2">
          {EXPORT_TYPES.map((t) => (
            <button key={t.id} onClick={() => setSelected(t.id)}
              className={`p-3 rounded-lg border text-left transition-colors ${
                selected === t.id
                  ? "bg-green-500/10 border-green-500/30"
                  : "bg-[var(--bg-input)] border-[var(--border)] hover:border-[var(--border-hi)]"
              }`}>
              <p className={`text-sm font-medium ${selected === t.id ? "text-green-400" : "text-[var(--text-base)]"}`}>
                {t.label}
              </p>
              <p className="text-xs text-[var(--text-dim)] mt-0.5">{t.desc}</p>
            </button>
          ))}
        </div>
        <button onClick={handleCreate} disabled={creating}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm font-medium rounded-lg transition-colors">
          {creating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
          {creating ? "Creating…" : "Create export job"}
        </button>
      </div>

      {/* Jobs list */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-[var(--text-base)]">Export history</h2>
          <button onClick={reload}
            className="p-1.5 rounded-lg border border-[var(--border)] text-[var(--text-dim)] hover:bg-[var(--bg-input)] transition-colors">
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-24">
            <div className="w-6 h-6 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-24 text-center">
            <Download className="w-8 h-8 text-[var(--text-ghost)] mb-2" />
            <p className="text-sm text-[var(--text-dim)]">No exports yet</p>
          </div>
        ) : (
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] text-[var(--text-dim)] text-xs">
                  <th className="px-4 py-3 text-left">Type</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-left">Created</th>
                  <th className="px-4 py-3 text-left">Download</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id} className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-input)]">
                    <td className="px-4 py-2.5 text-[var(--text-base)] font-mono text-xs">{job.export_type}</td>
                    <td className="px-4 py-2.5">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-4 py-2.5 text-[var(--text-muted)] text-xs">{formatDate(job.created_at)}</td>
                    <td className="px-4 py-2.5">
                      {job.status === "done" ? (
                        <a
                          href={`${API_BASE}/exports/${job.id}/download?user_id=${currentUserId}`}
                          download
                          className="flex items-center gap-1 text-xs text-green-500 hover:text-green-400 transition-colors"
                        >
                          <Download className="w-3.5 h-3.5" />
                          Download
                        </a>
                      ) : job.status === "running" ? (
                        <Loader2 className="w-3.5 h-3.5 text-[var(--text-muted)] animate-spin" />
                      ) : (
                        <span className="text-xs text-[var(--text-muted)]">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "done") return (
    <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] rounded border bg-green-500/10 text-green-400 border-green-500/20 w-fit">
      <CheckCircle className="w-3 h-3" /> done
    </span>
  );
  if (status === "running") return (
    <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] rounded border bg-yellow-500/10 text-yellow-400 border-yellow-500/20 w-fit">
      <Loader2 className="w-3 h-3 animate-spin" /> running
    </span>
  );
  return (
    <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] rounded border bg-red-500/10 text-red-400 border-red-500/20 w-fit">
      <AlertCircle className="w-3 h-3" /> {status}
    </span>
  );
}
