"use client";

import { useEffect, useState } from "react";
import { getImages, getTaxons, getDatasets, createDataset, deleteDataset } from "@/lib/api";
import type { ImageRecord, Taxon, Dataset } from "@/types";
import { formatDate } from "@/lib/utils";
import { Database, FolderOpen, Plus, Trash2, Images, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type Tab = "datasets" | "images" | "taxons";

export default function DatasetPanel() {
  const [activeTab, setActiveTab] = useState<Tab>("datasets");
  const [images, setImages]   = useState<ImageRecord[]>([]);
  const [taxons, setTaxons]   = useState<Taxon[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  const reload = async () => {
    setLoading(true);
    try {
      const [imgs, txs, dss] = await Promise.all([
        getImages({ status: "validated", size: 100 }),
        getTaxons(),
        getDatasets(),
      ]);
      setImages(imgs.items);
      setTaxons(txs);
      setDatasets(dss);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { reload(); }, []);

  const tabs: { id: Tab; label: string }[] = [
    { id: "datasets", label: `Datasets (${datasets.length})` },
    { id: "images",   label: `Validated (${images.length})` },
    { id: "taxons",   label: `Taxons (${taxons.length})` },
  ];

  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Dataset Explorer</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Manage your named datasets, validated images and taxonomy</p>
      </div>

      <div className="flex gap-1.5">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            className={cn(
              "px-3 py-1.5 text-xs rounded-lg border transition-colors",
              activeTab === t.id
                ? "bg-green-500/10 border-green-500/30 text-green-400"
                : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
            )}>
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : activeTab === "datasets" ? (
        <DatasetsTab datasets={datasets} onRefresh={reload} />
      ) : activeTab === "images" ? (
        <ImagesTab images={images} />
      ) : (
        <TaxonsTab taxons={taxons} />
      )}
    </div>
  );
}


// ── Datasets tab ─────────────────────────────────────────────────────────────

function DatasetsTab({ datasets, onRefresh }: { datasets: Dataset[]; onRefresh: () => void }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      await createDataset(name.trim(), desc.trim() || undefined);
      setName("");
      setDesc("");
      setShowForm(false);
      onRefresh();
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg || "Failed to create dataset");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: number) => {
    await deleteDataset(id);
    onRefresh();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-[var(--text-dim)]">{datasets.length} dataset{datasets.length !== 1 ? "s" : ""}</p>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-xs font-medium rounded-lg transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          New dataset
        </button>
      </div>

      {showForm && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
          <div>
            <label className="text-xs text-[var(--text-dim)] mb-1 block">Name *</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              placeholder="e.g. Ephemeroptera training set v1"
              className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors"
            />
          </div>
          <div>
            <label className="text-xs text-[var(--text-dim)] mb-1 block">Description (optional)</label>
            <input
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="e.g. Validated mayfly images from Wikimedia + iNaturalist"
              className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors"
            />
          </div>
          {error && <p className="text-xs text-red-400">{error}</p>}
          <div className="flex gap-2">
            <button onClick={handleCreate} disabled={creating || !name.trim()}
              className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg transition-colors">
              {creating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
              Create
            </button>
            <button onClick={() => setShowForm(false)}
              className="px-3 py-1.5 bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-dim)] text-xs rounded-lg hover:border-[var(--border-hi)] transition-colors">
              Cancel
            </button>
          </div>
        </div>
      )}

      {datasets.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 text-center">
          <FolderOpen className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
          <p className="text-sm text-[var(--text-dim)]">No datasets yet</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Create a named dataset to organise your validated images</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
          {datasets.map((ds) => (
            <div key={ds.id} className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 flex flex-col gap-3 hover:border-[var(--border-hi)] transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <div className="p-1.5 rounded-lg bg-indigo-500/10 shrink-0">
                    <FolderOpen className="w-4 h-4 text-indigo-400" />
                  </div>
                  <p className="text-sm font-medium text-[var(--text-base)] truncate">{ds.name}</p>
                </div>
                <button onClick={() => handleDelete(ds.id)}
                  className="p-1 text-[var(--text-muted)] hover:text-red-400 transition-colors shrink-0">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
              {ds.description && (
                <p className="text-xs text-[var(--text-dim)] line-clamp-2">{ds.description}</p>
              )}
              <div className="flex items-center justify-between mt-auto">
                <span className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
                  <Images className="w-3 h-3" />
                  {ds.image_count} image{ds.image_count !== 1 ? "s" : ""}
                </span>
                <span className="text-xs text-[var(--text-muted)]">{formatDate(ds.created_at)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// ── Images tab ───────────────────────────────────────────────────────────────

function ImagesTab({ images }: { images: ImageRecord[] }) {
  if (images.length === 0) return (
    <div className="flex flex-col items-center justify-center h-48 text-center">
      <Database className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
      <p className="text-sm text-[var(--text-dim)]">No validated images yet</p>
      <p className="text-xs text-[var(--text-muted)] mt-1">Validate images in the Validation Queue first</p>
    </div>
  );
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 xl:grid-cols-8 gap-2">
      {images.map((img) => (
        <div key={img.id} className="aspect-square rounded-lg overflow-hidden border border-[var(--border)] bg-[var(--bg-input)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={img.source_image_url} alt="" className="w-full h-full object-cover" loading="lazy" />
        </div>
      ))}
    </div>
  );
}


// ── Taxons tab ───────────────────────────────────────────────────────────────

function TaxonsTab({ taxons }: { taxons: Taxon[] }) {
  if (taxons.length === 0) return (
    <div className="flex flex-col items-center justify-center h-48 text-center">
      <Database className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
      <p className="text-sm text-[var(--text-dim)]">No taxons yet</p>
    </div>
  );
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--border)] text-[var(--text-dim)] text-xs">
            <th className="px-4 py-3 text-left">Scientific name</th>
            <th className="px-4 py-3 text-left">Common name</th>
            <th className="px-4 py-3 text-left">Rank</th>
            <th className="px-4 py-3 text-left">Added</th>
          </tr>
        </thead>
        <tbody>
          {taxons.map((t, i) => (
            <tr key={t.id} className={`border-b border-[var(--border)]/50 hover:bg-[var(--bg-input)] transition-colors ${i % 2 === 0 ? "" : "bg-[var(--bg-alt)]"}`}>
              <td className="px-4 py-2.5 text-[var(--text-base)] font-mono text-xs italic">{t.scientific_name}</td>
              <td className="px-4 py-2.5 text-[var(--text-dim)] text-xs">{t.common_name ?? "—"}</td>
              <td className="px-4 py-2.5 text-[var(--text-dim)] text-xs">{t.rank ?? "—"}</td>
              <td className="px-4 py-2.5 text-[var(--text-muted)] text-xs">{formatDate(t.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
