"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getImages, getTaxons, getDatasets, createDataset, deleteDataset, renameDataset,
  addImageToDataset, getDatasetImages, removeImageFromDataset,
  getAssignedImages, deleteImage, clearImages, uploadFiles,
} from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ImageRecord, Taxon, Dataset } from "@/types";
import { formatDate } from "@/lib/utils";
import {
  Database, FolderOpen, Plus, Trash2, Images, Loader2,
  ArrowLeft, X, ChevronDown, ChevronRight, AlertTriangle,
  EyeOff, FolderPlus, Check, Pencil, Upload, ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

type Tab = "datasets" | "images" | "taxons";
type AssignmentMap = Record<number, { dataset_id: number; dataset_name: string }>;
type TaxonGroup = { taxon: Taxon | null; taxonId: number | null; images: ImageRecord[] };

// ── Root panel ────────────────────────────────────────────────────────────────

export default function DatasetPanel() {
  const { currentUserId } = useAppStore();
  const [activeTab, setActiveTab] = useState<Tab>("datasets");
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [taxons, setTaxons] = useState<Taxon[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [assignments, setAssignments] = useState<AssignmentMap>({});
  const [loading, setLoading] = useState(true);

  const reload = async () => {
    if (!currentUserId) return;
    setLoading(true);
    try {
      const [imgs, txs, dss, asgn] = await Promise.all([
        getImages(currentUserId, { status: "validated", size: 500 }),
        getTaxons(currentUserId),
        getDatasets(currentUserId),
        getAssignedImages(currentUserId),
      ]);
      setImages(imgs.items);
      setTaxons(txs);
      setDatasets(dss);
      setAssignments(asgn);
    } catch { /* fail silently while backend boots */ }
    finally { setLoading(false); }
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
        <p className="text-sm text-[var(--text-dim)] mt-1">Manage your datasets, validated images and taxonomy</p>
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
        <DatasetsTab userId={currentUserId!} datasets={datasets} images={images} assignments={assignments} onRefresh={reload} />
      ) : activeTab === "images" ? (
        <ImagesTab userId={currentUserId!} images={images} taxons={taxons} datasets={datasets} assignments={assignments} onRefresh={reload} />
      ) : (
        <TaxonsTab taxons={taxons} />
      )}
    </div>
  );
}

// ── Shared helpers ─────────────────────────────────────────────────────────────

function groupByTaxon(imgs: ImageRecord[]): TaxonGroup[] {
  const map = new Map<string, TaxonGroup>();
  for (const img of imgs) {
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
}

// ── Confirm modal ─────────────────────────────────────────────────────────────

function ConfirmModal({ title, body, onConfirm, onCancel, loading }: {
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

// ── Datasets tab ───────────────────────────────────────────────────────────────

function DatasetsTab({ userId, datasets, images, assignments, onRefresh }: {
  userId: number;
  datasets: Dataset[];
  images: ImageRecord[];
  assignments: AssignmentMap;
  onRefresh: () => void;
}) {
  const { isReadOnly } = useAppStore();
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openDataset, setOpenDataset] = useState<Dataset | null>(null);
  const [dsImages, setDsImages] = useState<ImageRecord[]>([]);
  const [dsLoading, setDsLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [deletingDs, setDeletingDs] = useState(false);
  const [renamingId, setRenamingId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renaming, setRenaming] = useState(false);

  // Per-dataset species list derived from assignment map
  const datasetSpecies = useMemo(() => {
    const map = new Map<number, Map<number | null, Taxon | null>>();
    for (const img of images) {
      const a = assignments[img.id];
      if (!a) continue;
      if (!map.has(a.dataset_id)) map.set(a.dataset_id, new Map());
      if (!map.get(a.dataset_id)!.has(img.taxon_id))
        map.get(a.dataset_id)!.set(img.taxon_id, img.taxon);
    }
    return map;
  }, [images, assignments]);

  const openDetail = async (ds: Dataset) => {
    setOpenDataset(ds);
    setDsLoading(true);
    try {
      setDsImages(await getDatasetImages(userId, ds.id));
    } finally { setDsLoading(false); }
  };

  const handleRemoveTaxon = async (taxonId: number | null) => {
    if (!openDataset) return;
    const toRemove = dsImages.filter(img => img.taxon_id === taxonId);
    await Promise.all(toRemove.map(img => removeImageFromDataset(userId, openDataset.id, img.id)));
    setDsImages(prev => prev.filter(img => img.taxon_id !== taxonId));
    onRefresh();
  };

  const handleCreate = async () => {
    if (!name.trim()) return;
    setCreating(true); setError(null);
    try {
      await createDataset(userId, name.trim(), desc.trim() || undefined);
      setName(""); setDesc(""); setShowForm(false);
      onRefresh();
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg || "Failed to create dataset");
    } finally { setCreating(false); }
  };

  const handleDelete = async () => {
    if (!confirmDelete) return;
    setDeletingDs(true);
    try { await deleteDataset(userId, confirmDelete); setConfirmDelete(null); onRefresh(); }
    finally { setDeletingDs(false); }
  };

  const startRename = (ds: Dataset, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(ds.id);
    setRenameValue(ds.name);
  };

  const handleRename = async (dsId: number) => {
    if (!renameValue.trim()) return;
    setRenaming(true);
    try {
      await renameDataset(userId, dsId, renameValue.trim());
      setRenamingId(null);
      onRefresh();
    } finally { setRenaming(false); }
  };

  // ── Dataset detail view ──────────────────────────────────────────────────────
  if (openDataset) {
    const taxonGroups = groupByTaxon(dsImages);

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3 flex-wrap">
          <button onClick={() => { setOpenDataset(null); setDsImages([]); }}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg border bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)] transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" /> Back
          </button>
          <div className="flex items-center gap-2 min-w-0">
            <FolderOpen className="w-4 h-4 text-indigo-400 shrink-0" />
            <span className="text-sm font-semibold text-[var(--text-base)] truncate">{openDataset.name}</span>
          </div>
          <span className="text-xs text-[var(--text-muted)] ml-auto">
            {taxonGroups.length} species · {dsImages.length} images
          </span>
        </div>

        {dsLoading ? (
          <div className="flex items-center justify-center h-48">
            <div className="w-6 h-6 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : taxonGroups.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Images className="w-10 h-10 text-[var(--text-ghost)] mb-3" />
            <p className="text-sm text-[var(--text-dim)]">No images in this dataset yet</p>
            <p className="text-xs text-[var(--text-muted)] mt-1">Go to Validated tab and add species folders</p>
          </div>
        ) : (
          <div className="space-y-4">
            {taxonGroups.map((group) => {
              const key = group.taxonId ? String(group.taxonId) : "__none__";
              return (
                <div key={key} className="rounded-xl border border-[var(--border)] overflow-hidden">
                  {/* Species folder header */}
                  <div className="flex items-center gap-2 px-3 py-2.5 bg-[var(--bg-card)] border-b border-[var(--border)]">
                    <FolderOpen className="w-4 h-4 text-indigo-400 shrink-0" />
                    <span className="text-sm italic font-medium text-[var(--text-base)] truncate">
                      {group.taxon?.scientific_name ?? "Uncategorised"}
                    </span>
                    {group.taxon?.common_name && (
                      <span className="text-xs text-[var(--text-muted)] truncate hidden sm:inline">
                        — {group.taxon.common_name}
                      </span>
                    )}
                    <span className="px-1.5 py-0.5 text-[10px] rounded-full bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)] shrink-0">
                      {group.images.length}
                    </span>
                    {!isReadOnly && (
                      <button
                        onClick={() => handleRemoveTaxon(group.taxonId)}
                        className="ml-auto shrink-0 flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] text-red-400 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-colors"
                        title="Remove this species from dataset"
                      >
                        <X className="w-3 h-3" /> Remove species
                      </button>
                    )}
                  </div>
                  {/* Images */}
                  <div className="p-2 bg-[var(--bg-input)]/30 grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 xl:grid-cols-10 gap-1.5">
                    {group.images.map((img) => {
                      const src = img.local_path
                        ? `${API_BASE}/images/${img.id}/thumbnail?user_id=${img.user_id}`
                        : img.source_image_url;
                      return (
                        <div key={img.id} className="aspect-square rounded-lg overflow-hidden border border-[var(--border)]">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={src} alt="" className="w-full h-full object-cover bg-[var(--bg-input)]" loading="lazy" />
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  // ── Dataset list ──────────────────────────────────────────────────────────────
  return (
    <>
      {confirmDelete !== null && (
        <ConfirmModal
          title="Delete dataset?"
          body="The dataset and its organisation will be deleted. Validated images remain in your library."
          onConfirm={handleDelete}
          onCancel={() => setConfirmDelete(null)}
          loading={deletingDs}
        />
      )}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-[var(--text-dim)]">{datasets.length} dataset{datasets.length !== 1 ? "s" : ""}</p>
          {!isReadOnly && (
            <button onClick={() => setShowForm((v) => !v)}
              className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-xs font-medium rounded-lg transition-colors">
              <Plus className="w-3.5 h-3.5" /> New dataset
            </button>
          )}
        </div>

        {showForm && (
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
            <div>
              <label className="text-xs text-[var(--text-dim)] mb-1 block">Name *</label>
              <input value={name} onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                placeholder="e.g. Ephemeroptera training set v1" autoFocus
                className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors" />
            </div>
            <div>
              <label className="text-xs text-[var(--text-dim)] mb-1 block">Description (optional)</label>
              <input value={desc} onChange={(e) => setDesc(e.target.value)}
                placeholder="e.g. Validated mayfly images from Wikimedia + iNaturalist"
                className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors" />
            </div>
            {error && <p className="text-xs text-red-400">{error}</p>}
            <div className="flex gap-2">
              <button onClick={handleCreate} disabled={creating || !name.trim()}
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
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Create a dataset, then add species folders from the Validated tab
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {datasets.map((ds) => {
              const species = [...(datasetSpecies.get(ds.id)?.entries() ?? [])];
              const isRenaming = renamingId === ds.id;
              return (
                <div key={ds.id} onClick={() => !isRenaming && openDetail(ds)}
                  className={cn(
                    "bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 flex flex-col gap-3 transition-colors",
                    isRenaming ? "border-indigo-400/60" : "hover:border-indigo-400/50 cursor-pointer group/card"
                  )}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <div className="p-1.5 rounded-lg bg-indigo-500/10 shrink-0">
                        <FolderOpen className="w-4 h-4 text-indigo-400" />
                      </div>
                      {isRenaming ? (
                        <input
                          autoFocus
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleRename(ds.id);
                            if (e.key === "Escape") setRenamingId(null);
                          }}
                          onClick={(e) => e.stopPropagation()}
                          className="flex-1 min-w-0 bg-[var(--bg-input)] border border-indigo-400/40 rounded-lg px-2 py-1 text-sm text-[var(--text-base)] focus:outline-none"
                        />
                      ) : (
                        <p className="text-sm font-medium text-[var(--text-base)] truncate">{ds.name}</p>
                      )}
                    </div>
                    {!isReadOnly && (
                      <div className="flex items-center gap-1 shrink-0">
                        {isRenaming ? (
                          <>
                            <button onClick={(e) => { e.stopPropagation(); handleRename(ds.id); }} disabled={renaming}
                              className="p-1 text-green-400 hover:text-green-300 disabled:opacity-40 transition-colors">
                              {renaming ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                            </button>
                            <button onClick={(e) => { e.stopPropagation(); setRenamingId(null); }}
                              className="p-1 text-[var(--text-muted)] hover:text-[var(--text-dim)] transition-colors">
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </>
                        ) : (
                          <>
                            <button onClick={(e) => startRename(ds, e)}
                              className="p-1 text-[var(--text-muted)] hover:text-indigo-400 transition-colors opacity-0 group-hover/card:opacity-100">
                              <Pencil className="w-3.5 h-3.5" />
                            </button>
                            <button onClick={(e) => { e.stopPropagation(); setConfirmDelete(ds.id); }}
                              className="p-1 text-[var(--text-muted)] hover:text-red-400 transition-colors">
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                  {ds.description && (
                    <p className="text-xs text-[var(--text-dim)] line-clamp-2">{ds.description}</p>
                  )}
                  {/* Species badges */}
                  {species.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {species.slice(0, 4).map(([taxonId, taxon]) => (
                        <span key={taxonId ?? "none"}
                          className="px-1.5 py-0.5 text-[10px] rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 italic truncate max-w-[120px]">
                          {taxon?.scientific_name ?? "Uncategorised"}
                        </span>
                      ))}
                      {species.length > 4 && (
                        <span className="px-1.5 py-0.5 text-[10px] rounded bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)]">
                          +{species.length - 4} more
                        </span>
                      )}
                    </div>
                  )}
                  <div className="flex items-center justify-between mt-auto">
                    <span className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
                      <Images className="w-3 h-3" />
                      {ds.image_count} image{ds.image_count !== 1 ? "s" : ""}
                      {species.length > 0 && ` · ${species.length} species`}
                    </span>
                    <span className="text-xs text-[var(--text-muted)]">{formatDate(ds.created_at)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}

// ── Upload modal ──────────────────────────────────────────────────────────────

type Attribution = { source_url: string; author: string; license: string };

function parseAttrTxt(text: string): Attribution[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"))
    .map((line) => {
      const parts = line.split(" - ");
      const source_url = parts[0]?.trim() ?? "";
      const license = parts[parts.length - 1]?.trim() ?? "";
      const author = parts.length > 2 ? parts.slice(1, -1).join(" - ").trim() : (parts[1]?.trim() ?? "");
      return { source_url, author, license };
    });
}

function UploadModal({ userId, taxons, datasets, onDone, onClose }: {
  userId: number;
  taxons: Taxon[];
  datasets: Dataset[];
  onDone: () => void;
  onClose: () => void;
}) {
  const [images, setImages] = useState<File[]>([]);
  const [attrFileName, setAttrFileName] = useState<string | null>(null);
  const [attributions, setAttributions] = useState<Attribution[] | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [taxonMode, setTaxonMode] = useState<"existing" | "new">("existing");
  const [selectedTaxon, setSelectedTaxon] = useState<string>("");
  const [newSpeciesName, setNewSpeciesName] = useState("");
  const [targetDataset, setTargetDataset] = useState<number | "none">("none");
  const [imgDragging, setImgDragging] = useState(false);
  const [attrDragging, setAttrDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const imgRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const attrRef = useRef<HTMLInputElement>(null);

  const sortedImages = useMemo(
    () => [...images].sort((a, b) => a.name.localeCompare(b.name)),
    [images]
  );

  const addImageFiles = useCallback((fl: FileList | null) => {
    if (!fl) return;
    const imgs = Array.from(fl).filter((f) => f.type.startsWith("image/"));
    setImages((prev) => {
      const seen = new Set(prev.map((f) => f.name + f.size));
      return [...prev, ...imgs.filter((f) => !seen.has(f.name + f.size))];
    });
  }, []);

  const loadAttrFile = useCallback((fl: FileList | null) => {
    if (!fl) return;
    // From a folder pick: look for .txt file if mixed
    const txt = Array.from(fl).find((f) => f.name.toLowerCase().endsWith(".txt"));
    if (!txt) return;
    setAttrFileName(txt.name);
    txt.text().then((text) => {
      const parsed = parseAttrTxt(text);
      setAttributions(parsed);
      setParseError(null);
    });
  }, []);

  const handleFolderFromFileList = useCallback((fl: FileList | null) => {
    if (!fl) return;
    const all = Array.from(fl);
    const imgs = all.filter((f) => f.type.startsWith("image/"));
    const txt = all.find((f) => f.name.toLowerCase().endsWith(".txt"));
    if (imgs.length) addImageFiles(dataTransferFromArray(imgs));
    if (txt) loadAttrFile(dataTransferFromArray([txt]));
  }, [addImageFiles, loadAttrFile]);

  const handleFolderPicker = useCallback(async () => {
    // showDirectoryPicker requires HTTPS — fall back to webkitdirectory on plain HTTP
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (typeof window !== "undefined" && "showDirectoryPicker" in window) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const dirHandle = await (window as any).showDirectoryPicker({ mode: "read" });
        const imgs: File[] = [];
        let txtFile: File | null = null;
        for await (const entry of dirHandle.values()) {
          if (entry.kind !== "file") continue;
          const file: File = await entry.getFile();
          if (file.type.startsWith("image/")) imgs.push(file);
          else if (file.name.toLowerCase().endsWith(".txt") && !txtFile) txtFile = file;
        }
        if (imgs.length) addImageFiles(dataTransferFromArray(imgs));
        if (txtFile) loadAttrFile(dataTransferFromArray([txtFile]));
      } catch (err) {
        if ((err as Error).name !== "AbortError") console.error("Directory picker:", err);
      }
    } else {
      folderRef.current?.click();
    }
  }, [addImageFiles, loadAttrFile]);

  // Validate count match
  const countMismatch = attributions !== null && sortedImages.length > 0 && attributions.length !== sortedImages.length;
  const countOk = attributions !== null && sortedImages.length > 0 && attributions.length === sortedImages.length;

  const scientificName = taxonMode === "existing" ? selectedTaxon : newSpeciesName.trim();
  const canUpload = sortedImages.length > 0 && countOk && !!scientificName && !uploading;

  const handleUpload = async () => {
    if (!canUpload || !attributions) return;
    setUploading(true);
    setError(null);
    try {
      const uploaded = await uploadFiles(userId, sortedImages, scientificName, true, attributions);
      if (targetDataset !== "none" && uploaded.length) {
        await Promise.all(uploaded.map((img) => addImageToDataset(userId, targetDataset as number, img.id)));
      }
      onDone();
    } catch {
      setError("Upload failed — check the backend is running.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl w-full max-w-2xl shadow-2xl flex flex-col max-h-[92vh]">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)] shrink-0">
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-green-400" />
            <span className="text-sm font-semibold text-[var(--text-base)]">Upload images to Dataset Explorer</span>
          </div>
          <button onClick={onClose} className="p-1 text-[var(--text-muted)] hover:text-[var(--text-base)] transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-5 space-y-4">

          {/* ── Step 1 : Images ── */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-[var(--text-dim)]">
                1. Images <span className="text-red-400">*</span>
              </label>
              <div className="flex gap-1.5">
                <button onClick={() => imgRef.current?.click()}
                  className="flex items-center gap-1 px-2 py-1 text-[10px] rounded-lg border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)] hover:border-green-500/40 hover:text-green-400 transition-colors">
                  <ImageIcon className="w-3 h-3" /> Select files
                </button>
                <button onClick={handleFolderPicker}
                  className="flex items-center gap-1 px-2 py-1 text-[10px] rounded-lg border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)] hover:border-indigo-400/50 hover:text-indigo-300 transition-colors">
                  <FolderOpen className="w-3 h-3" /> Select folder
                </button>
                <input ref={imgRef} type="file" accept="image/*" multiple className="hidden"
                  onChange={(e) => addImageFiles(e.target.files)} />
                {/* @ts-expect-error webkitdirectory is non-standard — fallback when showDirectoryPicker unavailable (HTTP) */}
                <input ref={folderRef} type="file" webkitdirectory="" className="hidden"
                  onChange={(e) => handleFolderFromFileList(e.target.files)} />
              </div>
            </div>

            <div
              onDragOver={(e) => { e.preventDefault(); setImgDragging(true); }}
              onDragLeave={() => setImgDragging(false)}
              onDrop={(e) => { e.preventDefault(); setImgDragging(false); addImageFiles(e.dataTransfer.files); }}
              onClick={() => sortedImages.length === 0 && imgRef.current?.click()}
              className={cn(
                "border-2 border-dashed rounded-xl transition-colors",
                sortedImages.length === 0 ? "p-6 text-center cursor-pointer" : "p-0",
                imgDragging ? "border-green-500/50 bg-green-500/5" : "border-[var(--border)] hover:border-[var(--border-hi)]"
              )}
            >
              {sortedImages.length === 0 ? (
                <>
                  <ImageIcon className="w-6 h-6 text-[var(--text-ghost)] mx-auto mb-2" />
                  <p className="text-xs text-[var(--text-dim)]">Drop images here or use buttons above</p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">JPG · PNG · WebP · GIF — sorted alphabetically on upload</p>
                </>
              ) : (
                <div>
                  <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
                    <span className="text-xs text-[var(--text-dim)]">{sortedImages.length} image{sortedImages.length > 1 ? "s" : ""} (sorted alphabetically)</span>
                    <button onClick={(e) => { e.stopPropagation(); setImages([]); }}
                      className="text-[10px] text-[var(--text-muted)] hover:text-red-400 transition-colors">Clear</button>
                  </div>
                  <div className="max-h-28 overflow-y-auto divide-y divide-[var(--border)]/30">
                    {sortedImages.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 px-3 py-1">
                        <span className="text-[10px] text-[var(--text-muted)] w-5 shrink-0 text-right">{i + 1}</span>
                        <ImageIcon className="w-3 h-3 text-[var(--text-muted)] shrink-0" />
                        <span className="text-xs text-[var(--text-dim)] flex-1 truncate">{f.name}</span>
                        <span className="text-[10px] text-[var(--text-muted)] shrink-0">{(f.size / 1024).toFixed(0)} KB</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Step 2 : Attribution file ── */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-[var(--text-dim)]">
                2. Attribution file (.txt) <span className="text-red-400">*</span>
              </label>
              {attrFileName && (
                <button onClick={() => { setAttrFileName(null); setAttributions(null); }}
                  className="text-[10px] text-[var(--text-muted)] hover:text-red-400 transition-colors">Remove</button>
              )}
            </div>

            {!attrFileName ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setAttrDragging(true); }}
                onDragLeave={() => setAttrDragging(false)}
                onDrop={(e) => { e.preventDefault(); setAttrDragging(false); loadAttrFile(e.dataTransfer.files); }}
                onClick={() => attrRef.current?.click()}
                className={cn(
                  "border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-colors",
                  attrDragging ? "border-amber-500/50 bg-amber-500/5" : "border-[var(--border)] hover:border-[var(--border-hi)]"
                )}
              >
                <input ref={attrRef} type="file" accept=".txt" className="hidden"
                  onChange={(e) => loadAttrFile(e.target.files)} />
                <p className="text-xs text-[var(--text-dim)]">Drop any <code className="text-amber-400">.txt</code> attribution file here or click to select</p>
                <p className="text-[10px] text-[var(--text-muted)] mt-1 font-mono">
                  Format: source_url - author - license
                </p>
                <p className="text-[10px] text-[var(--text-muted)] mt-0.5 font-mono">
                  One line per image, in alphabetical order of filenames
                </p>
              </div>
            ) : (
              <div className={cn(
                "rounded-xl border overflow-hidden",
                countMismatch ? "border-red-500/40" : countOk ? "border-green-500/30" : "border-[var(--border)]"
              )}>
                <div className={cn(
                  "flex items-center gap-2 px-3 py-2 border-b text-xs",
                  countMismatch ? "bg-red-500/10 border-red-500/30 text-red-400"
                    : countOk ? "bg-green-500/10 border-green-500/30 text-green-400"
                    : "border-[var(--border)] text-[var(--text-dim)]"
                )}>
                  {countOk ? <Check className="w-3.5 h-3.5 shrink-0" /> : <AlertTriangle className="w-3.5 h-3.5 shrink-0" />}
                  <span className="font-medium truncate">{attrFileName}</span>
                  <span className="ml-auto shrink-0">
                    {attributions?.length ?? 0} line{attributions?.length !== 1 ? "s" : ""}
                    {sortedImages.length > 0 && ` / ${sortedImages.length} image${sortedImages.length !== 1 ? "s" : ""}`}
                  </span>
                </div>
                {countMismatch && (
                  <p className="px-3 py-2 text-[10px] text-red-400 bg-red-500/5">
                    Mismatch: {attributions?.length} attribution lines but {sortedImages.length} images. Each image needs exactly one line.
                  </p>
                )}
                {attributions && attributions.length > 0 && (
                  <div className="max-h-36 overflow-y-auto divide-y divide-[var(--border)]/30">
                    {attributions.map((a, i) => (
                      <div key={i} className="grid grid-cols-[2rem_1fr_auto_auto] items-center gap-2 px-3 py-1.5">
                        <span className="text-[10px] text-[var(--text-muted)] text-right">{i + 1}</span>
                        <span className="text-[10px] text-[var(--text-dim)] truncate font-mono">{a.source_url}</span>
                        <span className="text-[10px] text-[var(--text-muted)] truncate max-w-[80px]">{a.author}</span>
                        <span className="text-[10px] text-[var(--text-muted)] shrink-0">{a.license}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {parseError && <p className="text-[10px] text-red-400 mt-1">{parseError}</p>}
          </div>

          {/* ── Step 3 : Species folder ── */}
          <div>
            <label className="text-xs font-medium text-[var(--text-dim)] mb-2 block">
              3. Species folder <span className="text-red-400">*</span>
            </label>
            <div className="flex gap-2 mb-2">
              <button onClick={() => setTaxonMode("existing")}
                className={cn(
                  "flex-1 py-1.5 text-xs rounded-lg border transition-colors",
                  taxonMode === "existing"
                    ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-300"
                    : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
                )}>
                Existing folder
              </button>
              <button onClick={() => setTaxonMode("new")}
                className={cn(
                  "flex-1 py-1.5 text-xs rounded-lg border transition-colors",
                  taxonMode === "new"
                    ? "bg-green-500/10 border-green-500/30 text-green-400"
                    : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
                )}>
                New species
              </button>
            </div>
            {taxonMode === "existing" ? (
              taxons.length === 0 ? (
                <p className="text-xs text-[var(--text-muted)] px-1">No species folders yet — switch to New species.</p>
              ) : (
                <select value={selectedTaxon} onChange={(e) => setSelectedTaxon(e.target.value)}
                  className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] focus:outline-none focus:border-indigo-400/50 transition-colors">
                  <option value="">— choose a species —</option>
                  {taxons.map((t) => (
                    <option key={t.id} value={t.scientific_name}>
                      {t.scientific_name}{t.common_name ? ` — ${t.common_name}` : ""}
                    </option>
                  ))}
                </select>
              )
            ) : (
              <input autoFocus value={newSpeciesName} onChange={(e) => setNewSpeciesName(e.target.value)}
                placeholder="e.g. Baetis rhodani"
                className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm italic text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/40 transition-colors" />
            )}
          </div>

          {/* ── Step 4 : Dataset (optional) ── */}
          <div>
            <label className="text-xs font-medium text-[var(--text-dim)] mb-2 block">4. Add to dataset (optional)</label>
            <select value={targetDataset}
              onChange={(e) => setTargetDataset(e.target.value === "none" ? "none" : Number(e.target.value))}
              className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] focus:outline-none focus:border-indigo-400/50 transition-colors">
              <option value="none">No dataset — just add to Validated library</option>
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>{ds.name}</option>
              ))}
            </select>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-xs text-red-400">{error}</div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-[var(--border)] shrink-0 flex items-center justify-between gap-3">
          <p className="text-xs text-[var(--text-muted)]">
            Added directly as <span className="text-green-400">validated</span> — no validation queue.
          </p>
          <div className="flex gap-2 shrink-0">
            <button onClick={onClose}
              className="px-3 py-1.5 text-xs rounded-lg border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)] hover:border-[var(--border-hi)] transition-colors">
              Cancel
            </button>
            <button onClick={handleUpload} disabled={!canUpload}
              className="flex items-center gap-2 px-4 py-1.5 text-xs font-medium rounded-lg bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors">
              {uploading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Uploading…</>
                : <><Upload className="w-3.5 h-3.5" /> Upload {sortedImages.length > 0 ? `${sortedImages.length} image${sortedImages.length > 1 ? "s" : ""}` : ""}</>}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper: create a FileList-like object from an array of Files
function dataTransferFromArray(files: File[]): FileList {
  const dt = new DataTransfer();
  files.forEach((f) => dt.items.add(f));
  return dt.files;
}

// ── Images tab (Validated library) ────────────────────────────────────────────

type ConfirmTarget =
  | { kind: "image"; id: number; label: string }
  | { kind: "folder"; taxonId: number | null; label: string; count: number };

function AddToDatasetMenu({ group, datasets, assignments, userId, onRefresh }: {
  group: TaxonGroup;
  datasets: Dataset[];
  assignments: AssignmentMap;
  userId: number;
  onRefresh: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [adding, setAdding] = useState(false);
  const [creatingNew, setCreatingNew] = useState(false);
  const [newName, setNewName] = useState("");
  const menuRef = useRef<HTMLDivElement>(null);

  // Datasets this folder is already fully added to
  const addedDatasets = useMemo(() => {
    const counts = new Map<number, { name: string; count: number }>();
    for (const img of group.images) {
      const a = assignments[img.id];
      if (!a) continue;
      if (!counts.has(a.dataset_id)) counts.set(a.dataset_id, { name: a.dataset_name, count: 0 });
      counts.get(a.dataset_id)!.count++;
    }
    return counts;
  }, [group.images, assignments]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const handleAdd = async (datasetId: number) => {
    setAdding(true);
    try {
      const toAdd = group.images.filter(img => {
        const a = assignments[img.id];
        return !a || a.dataset_id !== datasetId;
      });
      await Promise.all(toAdd.map(img => addImageToDataset(userId, datasetId, img.id)));
      setOpen(false);
      onRefresh();
    } finally { setAdding(false); }
  };

  const handleCreateAndAdd = async () => {
    if (!newName.trim()) return;
    setAdding(true);
    try {
      const ds = await createDataset(userId, newName.trim());
      await Promise.all(group.images.map(img => addImageToDataset(userId, ds.id, img.id)));
      setOpen(false); setCreatingNew(false); setNewName("");
      onRefresh();
    } finally { setAdding(false); }
  };

  return (
    <div ref={menuRef} className="relative shrink-0">
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(v => !v); }}
        className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)] hover:border-indigo-400/50 hover:text-indigo-300 transition-colors"
        title="Add this species to a dataset"
      >
        {adding ? <Loader2 className="w-3 h-3 animate-spin" /> : <FolderPlus className="w-3 h-3" />}
        Add to dataset
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-52 bg-[var(--bg-card)] border border-[var(--border)] rounded-xl shadow-xl z-50 overflow-hidden">
          {datasets.length === 0 && !creatingNew ? (
            <p className="px-3 py-2 text-xs text-[var(--text-muted)]">No datasets yet — create one below</p>
          ) : (
            <div className="max-h-48 overflow-y-auto divide-y divide-[var(--border)]/40">
              {datasets.map((ds) => {
                const alreadyIn = addedDatasets.has(ds.id) && addedDatasets.get(ds.id)!.count === group.images.length;
                return (
                  <button key={ds.id}
                    onClick={() => !alreadyIn && handleAdd(ds.id)}
                    disabled={adding || alreadyIn}
                    className={cn(
                      "w-full flex items-center gap-2 px-3 py-2 text-left text-xs transition-colors",
                      alreadyIn
                        ? "text-[var(--text-muted)] cursor-default"
                        : "text-[var(--text-base)] hover:bg-[var(--bg-input)]"
                    )}>
                    <FolderOpen className={cn("w-3.5 h-3.5 shrink-0", alreadyIn ? "text-green-400" : "text-indigo-400")} />
                    <span className="truncate flex-1">{ds.name}</span>
                    {alreadyIn && <Check className="w-3 h-3 text-green-400 shrink-0" />}
                  </button>
                );
              })}
            </div>
          )}
          <div className="border-t border-[var(--border)] p-2">
            {creatingNew ? (
              <div className="space-y-1.5">
                <input
                  autoFocus
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleCreateAndAdd(); if (e.key === "Escape") { setCreatingNew(false); setNewName(""); } }}
                  placeholder="New dataset name…"
                  className="w-full bg-[var(--bg-input)] border border-green-500/40 rounded-lg px-2 py-1.5 text-xs text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none"
                />
                <div className="flex gap-1.5">
                  <button onClick={handleCreateAndAdd} disabled={adding || !newName.trim()}
                    className="flex-1 flex items-center justify-center gap-1 py-1 text-[10px] rounded-lg bg-green-600/20 hover:bg-green-600/40 text-green-400 disabled:opacity-40 transition-colors">
                    {adding ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                    Create & add
                  </button>
                  <button onClick={() => { setCreatingNew(false); setNewName(""); }}
                    className="px-2 py-1 text-[10px] rounded-lg bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)] transition-colors">
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button onClick={() => setCreatingNew(true)}
                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-[var(--text-dim)] hover:text-[var(--text-base)] hover:bg-[var(--bg-input)] rounded-lg transition-colors">
                <Plus className="w-3.5 h-3.5" /> New dataset…
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ImagesTab({ userId, images, taxons, datasets, assignments, onRefresh }: {
  userId: number;
  images: ImageRecord[];
  taxons: Taxon[];
  datasets: Dataset[];
  assignments: AssignmentMap;
  onRefresh: () => void;
}) {
  const { isReadOnly } = useAppStore();
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [confirmTarget, setConfirmTarget] = useState<ConfirmTarget | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [showUpload, setShowUpload] = useState(false);

  const groups = useMemo(() => groupByTaxon(images), [images]);

  const toggleCollapse = (key: string) =>
    setCollapsed((prev) => { const s = new Set(prev); s.has(key) ? s.delete(key) : s.add(key); return s; });

  const handleConfirmDelete = async () => {
    if (!confirmTarget) return;
    setDeleting(true);
    try {
      if (confirmTarget.kind === "image") {
        await deleteImage(userId, confirmTarget.id);
      } else {
        await clearImages(userId, "validated", confirmTarget.taxonId ?? undefined);
      }
      setConfirmTarget(null);
      onRefresh();
    } finally { setDeleting(false); }
  };

  return (
    <>
      {showUpload && (
        <UploadModal
          userId={userId}
          taxons={taxons}
          datasets={datasets}
          onDone={() => { setShowUpload(false); onRefresh(); }}
          onClose={() => setShowUpload(false)}
        />
      )}
      {confirmTarget && (
        <ConfirmModal
          title={confirmTarget.kind === "image" ? "Delete image?" : `Delete folder "${confirmTarget.label}"?`}
          body={confirmTarget.kind === "image"
            ? "This image will be permanently removed from your workspace."
            : `All ${confirmTarget.count} validated images in this species folder will be permanently deleted.`}
          onConfirm={handleConfirmDelete}
          onCancel={() => setConfirmTarget(null)}
          loading={deleting}
        />
      )}

      <div className="space-y-3">
        {/* Header bar */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-[var(--text-dim)]">
            {images.length} validated image{images.length !== 1 ? "s" : ""} · {groups.length} species folder{groups.length !== 1 ? "s" : ""}
          </p>
          {!isReadOnly && (
            <button onClick={() => setShowUpload(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)] hover:border-green-500/40 hover:text-green-400 transition-colors">
              <Upload className="w-3.5 h-3.5" /> Upload images
            </button>
          )}
        </div>

        {images.length === 0 && (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Database className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
            <p className="text-sm text-[var(--text-dim)]">No validated images yet</p>
            <p className="text-xs text-[var(--text-muted)] mt-1">Upload images above or validate them via the Validation Queue</p>
          </div>
        )}

        {isReadOnly && (
          <div className="flex items-center gap-2 px-3 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs text-amber-300">
            <EyeOff className="w-3.5 h-3.5 shrink-0" />
            <span>Read-only — login to manage images</span>
          </div>
        )}

        <div className="space-y-3">
          {groups.map((group) => {
            const key = group.taxonId ? String(group.taxonId) : "__none__";
            const isCollapsed = collapsed.has(key);

            // Which datasets contain images from this folder
            const inDatasets = [...new Map(
              group.images
                .filter(img => assignments[img.id])
                .map(img => [assignments[img.id].dataset_id, assignments[img.id].dataset_name])
            ).entries()];

            return (
              <div key={key} className="rounded-xl border border-[var(--border)] overflow-hidden">
                {/* Folder header */}
                <div className="flex items-center gap-2 px-3 py-2.5 bg-[var(--bg-card)] border-b border-[var(--border)]">
                  <button onClick={() => toggleCollapse(key)} className="flex items-center gap-2 flex-1 min-w-0 text-left">
                    {isCollapsed
                      ? <ChevronRight className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                      : <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />}
                    <FolderOpen className="w-4 h-4 text-amber-400 shrink-0" />
                    <span className="text-sm italic font-medium text-[var(--text-base)] truncate">
                      {group.taxon?.scientific_name ?? "Uncategorised"}
                    </span>
                    {group.taxon?.common_name && (
                      <span className="text-xs text-[var(--text-muted)] truncate hidden sm:inline">
                        — {group.taxon.common_name}
                      </span>
                    )}
                    {/* Dataset membership badges */}
                    {inDatasets.map(([dsId, dsName]) => (
                      <span key={dsId} className="hidden md:inline-block shrink-0 px-1.5 py-0.5 text-[9px] rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 italic truncate max-w-[100px]">
                        {dsName}
                      </span>
                    ))}
                    <span className="ml-auto shrink-0 px-1.5 py-0.5 text-[10px] rounded-full bg-[var(--bg-input)] border border-[var(--border)] text-[var(--text-muted)]">
                      {group.images.length}
                    </span>
                  </button>

                  {!isReadOnly && (
                    <>
                      <AddToDatasetMenu
                        group={group} datasets={datasets}
                        assignments={assignments} userId={userId} onRefresh={onRefresh}
                      />
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
                      const inDs = assignments[img.id];
                      return (
                        <div key={img.id} className="relative group/img aspect-square">
                          <div className={cn(
                            "absolute inset-0 rounded-lg overflow-hidden border",
                            inDs ? "border-indigo-400/40" : "border-[var(--border)]"
                          )}>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={src} alt="" className="w-full h-full object-cover bg-[var(--bg-input)]" loading="lazy" />
                            {inDs && (
                              <div className="absolute bottom-0 inset-x-0 bg-indigo-900/80 px-1 py-0.5 flex items-center gap-1">
                                <FolderOpen className="w-2 h-2 text-indigo-300 shrink-0" />
                                <span className="text-[8px] text-indigo-200 truncate">{inDs.dataset_name}</span>
                              </div>
                            )}
                          </div>
                          {!isReadOnly && (
                            <button
                              onClick={() => setConfirmTarget({ kind: "image", id: img.id, label: group.taxon?.scientific_name ?? "this image" })}
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

// ── Taxons tab ────────────────────────────────────────────────────────────────

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
