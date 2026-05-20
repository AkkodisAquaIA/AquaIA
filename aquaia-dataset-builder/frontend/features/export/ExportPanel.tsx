"use client";

import { useEffect, useState } from "react";
import { Download, Plus, Loader2 } from "lucide-react";
import { getExports, createExport } from "@/lib/api";
import type { ExportJob } from "@/types";
import { formatDate } from "@/lib/utils";

const EXPORT_TYPES = [
  { id: "classification", label: "Classification folders", desc: "class/image.jpg structure" },
  { id: "yolo", label: "YOLO", desc: "labels + images + data.yaml" },
  { id: "coco", label: "COCO JSON", desc: "instances_*.json format" },
  { id: "csv", label: "CSV", desc: "metadata spreadsheet" },
];

export default function ExportPanel() {
  const [jobs, setJobs] = useState<ExportJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [selected, setSelected] = useState("classification");

  useEffect(() => {
    getExports().then(setJobs).finally(() => setLoading(false));
  }, []);

  const handleCreate = async () => {
    setCreating(true);
    try {
      const job = await createExport(selected);
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
        <h2 className="text-sm font-semibold text-[var(--text-base)] mb-3">Export history</h2>
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
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id} className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-input)]">
                    <td className="px-4 py-2.5 text-[var(--text-base)] font-mono text-xs">{job.export_type}</td>
                    <td className="px-4 py-2.5">
                      <span className={`px-2 py-0.5 text-[10px] rounded border ${
                        job.status === "done"
                          ? "bg-green-500/10 text-green-400 border-green-500/20"
                          : "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
                      }`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-[var(--text-muted)] text-xs">{formatDate(job.created_at)}</td>
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
