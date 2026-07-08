"use client";

import { useEffect, useMemo, useState } from "react";
import { getImages, getTaxons, getDatasets, createDataset, deleteDataset, addImageToDataset, getDatasetImages, removeImageFromDataset, getAssignedImages, deleteImage, clearImages } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ImageRecord, Taxon, Dataset } from "@/types";
import { formatDate } from "@/lib/utils";
import { Database, FolderOpen, Plus, Trash2, Images, Loader2, CheckSquare, Square, FolderPlus, ArrowLeft, X, ChevronDown, ChevronRight, AlertTriangle, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";

const API_BASE_DS = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type Tab = "datasets" | "images" | "taxons";

type AssignmentMap = Record<number, { dataset_id: number; dataset_name: string }>;

export default function DatasetPanel() {
  const { currentUserId } = useAppStore();
  const [activeTab, setActiveTab] = useState<Tab>("datasets");
  const [images, setImages]   = useState<ImageRecord[]>([]);
  const [taxons, setTaxons]   = useState<Taxon[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [assignments, setAssignments] = useState<AssignmentMap>({});
  const [loading, setLoading] = useState(true);

  const reload = async () => {
    if (!currentUserId) return;
    setLoading(true);
    try {
      const [imgs, txs, dss, asgn] = await Promise.all([
        getImages(currentUserId, { status: "validated", size: 100 }),
        getTaxons(currentUserId),
        getDatasets(currentUserId),
        getAssignedImages(currentUserId),
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

  useEffect(() => { reload(); }, [currentUserId]);

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
        <DatasetsTab userId={currentUserId!} datasets={datasets} onRefresh={reload} />
      ) : activeTab === "images" ? (
        <ImagesTab userId={currentUserId!} images={images} datasets={datasets} assignments={assignments} onRefresh={reload} />
      ) : (
        <TaxonsTab taxons={taxons} />
      )}
    </div>
  );
}


// ── Datasets tab ─────────────────────────────────────────────────────────────

function DatasetsTab({ userId, datasets, onRefresh }: { userId: number; datasets: Dataset[]; onRefresh: () => void }) {
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
      const imgs = await getDatasetImages(userId, ds.id);
      setDsImages(imgs);
    } finally {
      setDsLoading(false);
    }
  };

  const handleRemoveImage = async (imageId: number) => {
    if (!openDataset) return;
    await removeImageFromDataset(userId, openDataset.id, imageId);
    setDsImages((prev) => prev.filter((i) => i.id !== imageId));
    onRefresh();
  };

  const handleCreate = async () => {
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      await createDataset(userId, name.trim(), desc.trim() || undefined);
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
    await deleteDataset(userId, id);
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
              ? `${API_BASE_DS}/images/${img.id}/thumbnail?user_id=${img.user_id}`
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


// ── Confirm modal ─────────────────────────────────────────────────────────────

function ConfirmModal({
  title, body, onConfirm, onCancel, loading,
}: {
  title: string; body: string; onConfirm: () => void; onCancel: () => void; loading?: boolean;
}) {
  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-6 w-full max-w-sm shadow-2xl mx-4">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-xl bg-red-500/10 border border-red-500/20 shrink-0">
            <AlertTriangle className="w-5 h-5 text-red-400" />
          </div>
          <h3 className="text-base font-semibold text-[var(--text-base)]">{title}</h3>
        </div>
        <p className="text-sm text-[var(--text-dim)] mb-5">{body}</p>
        <div className="flex gap-3">
          <button onClick={onCancel}
            className="flex-1 px-4 py-2 rounded-xl border border-[var(--border)] bg-[var(--bg-input)] text-sm text-[var(--text-dim)] hover:border-[var(--border-hi)] transition-colors">
            Cancel
          </button>
          <button onClick={onConfirm} disabled={loading}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 disabled:opacity-40 text-white text-sm font-medium transition-colors">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Images tab ───────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type TaxonGroup = { taxon: Taxon | null; taxonId: number | null; images: ImageRecord[] };
type ConfirmTarget =
  | { kind: "image"; id: number; label: string }
  | { kind: "folder"; taxonId: number | null; label: string; count: number };

function ImagesTab({
  userId, images, datasets, assignments, onRefresh,
}: {
  userId: number;
  images: ImageRecord[];
  datasets: Dataset[];
  assignments: AssignmentMap;
  onRefresh: () => void;
}) {
  const { isReadOnly } = useAppStore();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [targetDataset, setTargetDataset] = useState<number | "">("");
  const [adding, setAdding] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [confirmTarget, setConfirmTarget] = useState<ConfirmTarget | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Group images by taxon, sorted alphabetically (uncategorised last)
  const groups = useMemo<TaxonGroup[]>(() => {
    const map = new Map<string, TaxonGroup>();
    for (const img of images) {
      const key = img.taxon_id ? String(img.taxon_id) : "__none__";
      if (!map.has(key)) map.set(key, { taxon: img.taxon, taxonId: img.taxon_id, images: [] });
      map.get(key)!.images.push(img);
    }
    return [...map.entries()]
      .sort(([ka, a], [kb, b]) => {
        if (ka === "__none__") return 1;
        if (kb === "__none__") return -1;
        return (a.taxon?.scientific_name ?? "").localeCompare(b.taxon?.scientific_name ?? "");
      })
      .map(([, v]) => v);
  }, [images]);

  const toggle = (id: number) =>
    setSelected((prev) => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const toggleGroup = (key: string, groupImages: ImageRecord[]) => {
    const ids = groupImages.map((i) => i.id);
    const allSelected = ids.every((id) => selected.has(id));
    setSelected((prev) => {
      const s = new Set(prev);
      if (allSelected) ids.forEach((id) => s.delete(id));
      else ids.forEach((id) => s.add(id));
      return s;
    });
  };

  const selectAll = () => setSelected(new Set(images.map((i) => i.id)));
  const clearAll  = () => setSelected(new Set());

  const toggleCollapse = (key: string) =>
    setCollapsed((prev) => { const s = new Set(prev); s.has(key) ? s.delete(key) : s.add(key); return s; });

  const handleAddToDataset = async () => {
    if (!targetDataset || selected.size === 0) return;
    const targetId = Number(targetDataset);
    setWarning(null);

    const alreadyHere: number[] = [];
    const alreadyElsewhere: { id: number; dsName: string }[] = [];
    const toAdd: number[] = [];

    for (const imgId of selected) {
      const a = assignments[imgId];
      if (!a) toAdd.push(imgId);
      else if (a.dataset_id === targetId) alreadyHere.push(imgId);
      else alreadyElsewhere.push({ id: imgId, dsName: a.dataset_name });
    }

    if (alreadyElsewhere.length > 0) {
      const names = [...new Set(alreadyElsewhere.map((e) => e.dsName))].join(", ");
      setWarning(`${alreadyElsewhere.length} image(s) already assigned to: ${names}. Only unassigned images were added.`);
    }
    if (toAdd.length === 0) {
      if (alreadyHere.length > 0 && alreadyElsewhere.length === 0)
        setWarning("All selected images are already in this dataset.");
      return;
    }

    setAdding(true);
    try {
      await Promise.all(toAdd.map((imgId) => addImageToDataset(userId, targetId, imgId)));
      setSelected(new Set());
      setTargetDataset("");
      onRefresh();
    } finally {
      setAdding(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!confirmTarget) return;
    setDeleting(true);
    try {
      if (confirmTarget.kind === "image") {
        await deleteImage(userId, confirmTarget.id);
      } else {
        await clearImages(userId, "validated", confirmTarget.taxonId ?? undefined);
      }
      setSelected((prev) => {
        const s = new Set(prev);
        if (confirmTarget.kind === "image") s.delete(confirmTarget.id);
        return s;
      });
      setConfirmTarget(null);
      onRefresh();
    } finally {
      setDeleting(false);
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
    <>
    {confirmTarget && (
      <ConfirmModal
        title={confirmTarget.kind === "image" ? "Delete image?" : `Delete folder "${confirmTarget.label}"?`}
        body={confirmTarget.kind === "image"
          ? "This image will be permanently removed from your workspace."
          : `All ${confirmTarget.count} validated images in this taxon folder will be permanently deleted.`}
        onConfirm={handleConfirmDelete}
        onCancel={() => setConfirmTarget(null)}
        loading={deleting}
      />
    )}
    <div className="space-y-3">
      {/* Read-only banner */}
      {isReadOnly && (
        <div className="flex items-center gap-2 px-3 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs text-amber-300">
          <EyeOff className="w-3.5 h-3.5 shrink-0" />
          <span>Read-only — login to manage images</span>
        </div>
      )}

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

      {/* Action bar — hidden in read-only mode */}
      {!isReadOnly && <div className="flex items-center gap-3 flex-wrap">
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
      </div>}

      {/* Taxon groups */}
      <div className="space-y-4">
        {groups.map((group) => {
          const key = group.taxonId ? String(group.taxonId) : "__none__";
          const isCollapsed = collapsed.has(key);
          const groupIds = group.images.map((i) => i.id);
          const allGroupSelected = groupIds.every((id) => selected.has(id));
          const someGroupSelected = groupIds.some((id) => selected.has(id));

          return (
            <div key={key} className="rounded-xl border border-[var(--border)] overflow-hidden">
              {/* Folder header */}
              <div className="flex items-center gap-2 px-3 py-2.5 bg-[var(--bg-card)] border-b border-[var(--border)]">
                <button
                  onClick={() => toggleCollapse(key)}
                  className="flex items-center gap-2 flex-1 min-w-0 text-left"
                >
                  {isCollapsed
                    ? <ChevronRight className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                    : <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />}
                  <FolderOpen className="w-4 h-4 text-amber-400 shrink-0" />
                  <span className="text-sm italic text-[var(--text-base)] font-medium truncate">
                    {group.taxon?.scientific_name ?? "Uncategorised"}
                  </span>
                  {group.taxon?.common_name && (
                    <span className="text-xs text-[var(--text-muted)] truncate hidden sm:inline">
                      — {group.taxon.common_name}
                    </span>
                  )}
                  <span className="ml-auto shrink-0 px-1.5 py-0.5 text-[10px] rounded-full bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)]">
                    {group.images.length}
                  </span>
                </button>
                {!isReadOnly && (
                  <>
                    {/* Select all in group */}
                    <button
                      onClick={() => toggleGroup(key, group.images)}
                      className="shrink-0 p-1 rounded hover:bg-[var(--bg-input)] transition-colors"
                      title={allGroupSelected ? "Deselect folder" : "Select all in folder"}
                    >
                      {allGroupSelected
                        ? <CheckSquare className="w-3.5 h-3.5 text-green-400" />
                        : someGroupSelected
                        ? <Square className="w-3.5 h-3.5 text-green-400/50" />
                        : <Square className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
                    </button>
                    {/* Delete folder */}
                    <button
                      onClick={() => setConfirmTarget({ kind: "folder", taxonId: group.taxonId, label: group.taxon?.scientific_name ?? "Uncategorised", count: group.images.length })}
                      className="shrink-0 p-1 rounded hover:bg-red-500/20 text-[var(--text-muted)] hover:text-red-400 transition-colors"
                      title="Delete all images in this folder"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </div>

              {/* Images grid */}
              {!isCollapsed && (
                <div className="p-2 bg-[var(--bg-input)]/30 grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 xl:grid-cols-10 gap-1.5">
                  {group.images.map((img) => {
                    const src = img.local_path
                      ? `${API_BASE}/images/${img.id}/thumbnail?user_id=${img.user_id}`
                      : img.source_image_url;
                    const isSelected = selected.has(img.id);
                    const assignment = assignments[img.id];
                    return (
                      <div key={img.id} className="relative group/img aspect-square">
                      <button onClick={() => !isReadOnly && toggle(img.id)}
                        className={cn(
                          "absolute inset-0 rounded-lg overflow-hidden border transition-all w-full h-full",
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
                            <div className="w-3.5 h-3.5 rounded-full bg-green-500 flex items-center justify-center">
                              <CheckSquare className="w-2 h-2 text-white" />
                            </div>
                          </div>
                        )}
                        {assignment && !isSelected && (
                          <div className="absolute bottom-0 inset-x-0 bg-indigo-900/80 px-1 py-0.5 flex items-center gap-1">
                            <FolderOpen className="w-2 h-2 text-indigo-300 shrink-0" />
                            <span className="text-[8px] text-indigo-200 truncate leading-tight">{assignment.dataset_name}</span>
                          </div>
                        )}
                      </button>
                      {/* Delete single image — visible on hover, hidden in read-only */}
                      {!isReadOnly && (
                        <button
                          onClick={(e) => { e.stopPropagation(); setConfirmTarget({ kind: "image", id: img.id, label: group.taxon?.scientific_name ?? "this image" }); }}
                          className="absolute top-1 left-1 w-5 h-5 rounded bg-red-600/80 hover:bg-red-500 flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-opacity z-10"
                          title="Delete image"
                        >
                          <Trash2 className="w-3 h-3 text-white" />
                        </button>
                      )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
    </>
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
