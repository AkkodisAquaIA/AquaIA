"use client";

import { useEffect, useState } from "react";
import { getImages, getTaxons, getDatasets, createDataset, deleteDataset, addImageToDataset, getDatasetImages, removeImageFromDataset, getAssignedImages } from "@/lib/api";
import type { ImageRecord, Taxon, Dataset } from "@/types";
import { formatDate } from "@/lib/utils";
import { Database, FolderOpen, Plus, Trash2, Images, Loader2, CheckSquare, Square, FolderPlus, ArrowLeft, X } from "lucide-react";
import { cn } from "@/lib/utils";

const API_BASE_DS = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type Tab = "datasets" | "images" | "taxons";

type AssignmentMap = Record<number, { dataset_id: number; dataset_name: string }>;

export default function DatasetPanel() {
  const [activeTab, setActiveTab] = useState<Tab>("datasets");
  const [images, setImages]   = useState<ImageRecord[]>([]);
  const [taxons, setTaxons]   = useState<Taxon[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [assignments, setAssignments] = useState<AssignmentMap>({});
  const [loading, setLoading] = useState(true);

  const reload = async () => {
    setLoading(true);
    try {
      const [imgs, txs, dss, asgn] = await Promise.all([
        getImages({ status: "validated", size: 100 }),
        getTaxons(),
        getDatasets(),
        getAssignedImages(),
      ]);
      setImages(imgs.items);
      setTaxons(txs);
      setDatasets(dss);
      setAssignments(asgn);
    } catch {
      // Backend may not be fully up yet — fail silently
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
        <ImagesTab images={images} datasets={datasets} assignments={assignments} onRefresh={reload} />
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
  const [openDataset, setOpenDataset] = useState<Dataset | null>(null);
  const [dsImages, setDsImages] = useState<ImageRecord[]>([]);
  const [dsLoading, setDsLoading] = useState(false);

  const openDetail = async (ds: Dataset) => {
    setOpenDataset(ds);
    setDsLoading(true);
    try {
      const imgs = await getDatasetImages(ds.id);
      setDsImages(imgs);
    } finally {
      setDsLoading(false);
    }
  };

  const handleRemoveImage = async (imageId: number) => {
    if (!openDataset) return;
    await removeImageFromDataset(openDataset.id, imageId);
    setDsImages((prev) => prev.filter((i) => i.id !== imageId));
    onRefresh();
  };

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

  // ── Dataset detail view ──────────────────────────────────────────────────
  if (openDataset) return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button onClick={() => { setOpenDataset(null); setDsImages([]); }}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg border bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)] transition-colors">
          <ArrowLeft className="w-3.5 h-3.5" /> Back
        </button>
        <div className="flex items-center gap-2">
          <FolderOpen className="w-4 h-4 text-indigo-400" />
          <span className="text-sm font-semibold text-[var(--text-base)]">{openDataset.name}</span>
          <span className="text-xs text-[var(--text-muted)]">— {dsImages.length} image{dsImages.length !== 1 ? "s" : ""}</span>
        </div>
      </div>

      {dsLoading ? (
        <div className="flex items-center justify-center h-48">
          <div className="w-6 h-6 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : dsImages.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 text-center">
          <Images className="w-10 h-10 text-[var(--text-ghost)] mb-3" />
          <p className="text-sm text-[var(--text-dim)]">No images in this dataset</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Go to the Validated tab to add images</p>
        </div>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 xl:grid-cols-8 gap-2">
          {dsImages.map((img) => {
            const src = img.local_path
              ? `${API_BASE_DS}/images/${img.id}/thumbnail`
              : img.source_image_url;
            return (
              <div key={img.id} className="group relative aspect-square rounded-lg overflow-hidden border border-[var(--border)] bg-[var(--bg-input)]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={src} alt="" className="w-full h-full object-cover" loading="lazy" />
                <button
                  onClick={() => handleRemoveImage(img.id)}
                  className="absolute top-1 right-1 p-1 bg-black/60 rounded-md opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/80"
                  title="Remove from dataset">
                  <X className="w-3 h-3 text-white" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  // ── Dataset list ──────────────────────────────────────────────────────────
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
              autoFocus
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
              title={!name.trim() ? "Enter a name first" : undefined}
              className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium rounded-lg transition-colors">
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
            <div key={ds.id} onClick={() => openDetail(ds)}
              className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 flex flex-col gap-3 hover:border-indigo-400/50 cursor-pointer transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <div className="p-1.5 rounded-lg bg-indigo-500/10 shrink-0">
                    <FolderOpen className="w-4 h-4 text-indigo-400" />
                  </div>
                  <p className="text-sm font-medium text-[var(--text-base)] truncate">{ds.name}</p>
                </div>
                <button onClick={(e) => { e.stopPropagation(); handleDelete(ds.id); }}
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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

function ImagesTab({
  images, datasets, assignments, onRefresh,
}: {
  images: ImageRecord[];
  datasets: Dataset[];
  assignments: AssignmentMap;
  onRefresh: () => void;
}) {
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [targetDataset, setTargetDataset] = useState<number | "">("");
  const [adding, setAdding] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);

  const toggle = (id: number) =>
    setSelected((prev) => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const selectAll = () => setSelected(new Set(images.map((i) => i.id)));
  const clearAll  = () => setSelected(new Set());

  const handleAddToDataset = async () => {
    if (!targetDataset || selected.size === 0) return;
    const targetId = Number(targetDataset);
    setWarning(null);

    const alreadyHere: number[]  = [];
    const alreadyElsewhere: { id: number; dsName: string }[] = [];
    const toAdd: number[] = [];

    for (const imgId of selected) {
      const a = assignments[imgId];
      if (!a) {
        toAdd.push(imgId);
      } else if (a.dataset_id === targetId) {
        alreadyHere.push(imgId);
      } else {
        alreadyElsewhere.push({ id: imgId, dsName: a.dataset_name });
      }
    }

    if (alreadyElsewhere.length > 0) {
      const names = [...new Set(alreadyElsewhere.map((e) => e.dsName))].join(", ");
      const count = alreadyElsewhere.length;
      setWarning(
        `${count} image${count > 1 ? "s" : ""} already assigned to: ${names}. Only unassigned images were added.`
      );
    }

    if (toAdd.length === 0) {
      if (alreadyHere.length > 0 && alreadyElsewhere.length === 0) {
        setWarning("All selected images are already in this dataset.");
      }
      return;
    }

    setAdding(true);
    try {
      await Promise.all(toAdd.map((imgId) => addImageToDataset(targetId, imgId)));
      setSelected(new Set());
      setTargetDataset("");
      onRefresh();
    } finally {
      setAdding(false);
    }
  };

  if (images.length === 0) return (
    <div className="flex flex-col items-center justify-center h-48 text-center">
      <Database className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
      <p className="text-sm text-[var(--text-dim)]">No validated images yet</p>
      <p className="text-xs text-[var(--text-muted)] mt-1">Validate images in the Validation Queue first</p>
    </div>
  );

  return (
    <div className="space-y-3">
      {/* Warning banner */}
      {warning && (
        <div className="flex items-start gap-2 px-3 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs text-amber-300">
          <span className="shrink-0 mt-0.5">⚠</span>
          <span>{warning}</span>
          <button onClick={() => setWarning(null)} className="ml-auto shrink-0 opacity-60 hover:opacity-100">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Action bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <button onClick={selected.size === images.length ? clearAll : selectAll}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg border bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)] transition-colors">
            {selected.size === images.length
              ? <CheckSquare className="w-3.5 h-3.5 text-green-400" />
              : <Square className="w-3.5 h-3.5" />}
            {selected.size === images.length ? "Deselect all" : "Select all"}
          </button>
          {selected.size > 0 && (
            <span className="text-xs text-green-400 font-medium">{selected.size} selected</span>
          )}
        </div>

        {selected.size > 0 && (
          <div className="flex items-center gap-2 ml-auto">
            <FolderPlus className="w-4 h-4 text-[var(--text-muted)]" />
            <select
              value={targetDataset}
              onChange={(e) => { setTargetDataset(e.target.value ? Number(e.target.value) : ""); setWarning(null); }}
              className="bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-base)] focus:outline-none focus:border-green-500/50"
            >
              <option value="">Choose a dataset…</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
            <button
              onClick={handleAddToDataset}
              disabled={!targetDataset || adding}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg transition-colors"
            >
              {adding ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
              Add to dataset
            </button>
          </div>
        )}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 xl:grid-cols-8 gap-2">
        {images.map((img) => {
          const src = img.local_path
            ? `${API_BASE}/images/${img.id}/thumbnail`
            : img.source_image_url;
          const isSelected = selected.has(img.id);
          const assignment = assignments[img.id];
          return (
            <button key={img.id} onClick={() => toggle(img.id)}
              className={cn(
                "relative aspect-square rounded-lg overflow-hidden border transition-all",
                isSelected
                  ? "border-green-500 ring-2 ring-green-500/30"
                  : assignment
                  ? "border-indigo-400/50"
                  : "border-[var(--border)] hover:border-[var(--border-hi)]"
              )}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src} alt="" className="w-full h-full object-cover bg-[var(--bg-input)]" loading="lazy" />
              {isSelected && (
                <div className="absolute inset-0 bg-green-500/20 flex items-start justify-end p-1">
                  <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center">
                    <CheckSquare className="w-2.5 h-2.5 text-white" />
                  </div>
                </div>
              )}
              {assignment && !isSelected && (
                <div className="absolute bottom-0 inset-x-0 bg-indigo-900/80 px-1 py-0.5 flex items-center gap-1">
                  <FolderOpen className="w-2.5 h-2.5 text-indigo-300 shrink-0" />
                  <span className="text-[9px] text-indigo-200 truncate leading-tight">{assignment.dataset_name}</span>
                </div>
              )}
            </button>
          );
        })}
      </div>
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
