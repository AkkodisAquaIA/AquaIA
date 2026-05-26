"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Check, X, Copy, Clock, ExternalLink, ChevronLeft, ChevronRight, Maximize2, Crop, RotateCcw, BookImage, Star } from "lucide-react";
import { getImages, updateImageStatus, cropImage, setTaxonReference } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ImageRecord } from "@/types";
import { cn } from "@/lib/utils";
import ReactCrop, { type Crop as CropType, centerCrop, makeAspectCrop } from "react-image-crop";
import "react-image-crop/dist/ReactCrop.css";

const FILTERS = [
  { id: "pending",      label: "Pending" },
  { id: "validated",    label: "Validated" },
  { id: "rejected",     label: "Rejected" },
  { id: "duplicate",    label: "Duplicates" },
  { id: "review_later", label: "Later" },
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

function imgSrc(image: ImageRecord, bust?: number): string {
  if (image.local_path)
    return `${API_BASE}/images/${image.id}/file${bust ? `?t=${bust}` : ""}`;
  return image.source_image_url;
}

export default function ValidationPanel() {
  const { validationFilter, setValidationFilter, cropWidth, cropHeight } = useAppStore();
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<ImageRecord | null>(null);
  const [lightbox, setLightbox] = useState(false);
  const [cropMode, setCropMode] = useState(false);
  const [crop, setCrop] = useState<CropType>();
  const [saving, setSaving] = useState(false);
  const [imgBust, setImgBust] = useState(0);
  const [settingRef, setSettingRef] = useState(false);
  const [refPanelHeight, setRefPanelHeight] = useState(160);
  const dragState = useRef<{ startY: number; startH: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleDividerMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    dragState.current = { startY: e.clientY, startH: refPanelHeight };
    const onMove = (ev: MouseEvent) => {
      if (!dragState.current) return;
      const delta = dragState.current.startY - ev.clientY;
      setRefPanelHeight(Math.max(80, Math.min(420, dragState.current.startH + delta)));
    };
    const onUp = () => {
      dragState.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getImages({ status: validationFilter, page, size: 48 });
      setImages(data.items);
      setTotal(data.total);
      setPages(data.pages);
    } catch {
      // Backend may not be up yet — fail silently
    } finally {
      setLoading(false);
    }
  }, [validationFilter, page]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { setPage(1); }, [validationFilter]);

  const handleStatus = async (id: number, status: string) => {
    await updateImageStatus(id, status);
    const leavesFilter = status !== validationFilter;
    setImages((prev) => {
      const idx = prev.findIndex((img) => img.id === id);
      const next = prev[idx + 1] ?? prev[idx - 1] ?? null;
      setSelected(next);
      return leavesFilter ? prev.filter((img) => img.id !== id) : prev;
    });
    if (leavesFilter) setTotal((t) => t - 1);
  };

  // Keyboard shortcuts
  useEffect(() => {
    if (!selected) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setCropMode(false); setLightbox(false); return; }
      if (lightbox) return;
      if (e.key === "v" || e.key === "V") handleStatus(selected.id, "validated");
      if (e.key === "r" || e.key === "R") handleStatus(selected.id, "rejected");
      if (e.key === "d" || e.key === "D") handleStatus(selected.id, "duplicate");
      if (e.key === " ") { e.preventDefault(); setSelected(null); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, lightbox]);

  return (
    <div className="panel-enter flex gap-4 h-full">
      {/* Left: grid */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-[var(--text-base)]">Validation Queue</h1>
            <p className="text-sm text-[var(--text-dim)] mt-0.5">{total} images</p>
          </div>
          <div className="flex gap-1.5">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                onClick={() => setValidationFilter(f.id)}
                className={cn(
                  "px-3 py-1.5 text-xs rounded-lg border transition-colors",
                  validationFilter === f.id
                    ? "bg-green-500/10 border-green-500/30 text-green-400"
                    : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-48">
            <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : images.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Check className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
            <p className="text-sm text-[var(--text-dim)]">No images in this category</p>
          </div>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 xl:grid-cols-6 gap-2 overflow-y-auto">
            {images.map((img) => (
              <button
                key={img.id}
                onClick={() => setSelected(img)}
                className={cn(
                  "group relative aspect-square rounded-lg overflow-hidden border transition-all",
                  selected?.id === img.id
                    ? "border-green-500 ring-1 ring-green-500/30"
                    : "border-[var(--border)] hover:border-[var(--border-hi)]"
                )}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={img.source_image_url}
                  alt=""
                  className="w-full h-full object-cover bg-[var(--bg-input)]"
                  loading="lazy"
                />
              </button>
            ))}
          </div>
        )}

        {/* Pagination */}
        {pages > 1 && (
          <div className="flex items-center gap-2 justify-end pt-2">
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1}
              className="p-1.5 rounded-lg border border-[var(--border)] text-[var(--text-dim)] disabled:opacity-30 hover:bg-[var(--bg-input)]">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs text-[var(--text-dim)]">{page} / {pages}</span>
            <button onClick={() => setPage((p) => Math.min(pages, p + 1))} disabled={page === pages}
              className="p-1.5 rounded-lg border border-[var(--border)] text-[var(--text-dim)] disabled:opacity-30 hover:bg-[var(--bg-input)]">
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Lightbox overlay */}
      {lightbox && selected && (
        <div className="fixed inset-0 z-50 bg-black/90 flex flex-col items-center justify-center gap-4"
          onClick={() => { if (!cropMode) { setLightbox(false); setCropMode(false); setCrop(undefined); } }}>

          {/* Toolbar */}
          <div className="flex items-center gap-2 z-10" onClick={(e) => e.stopPropagation()}>
            {!cropMode ? (
              <button onClick={() => {
                setCropMode(true);
                // Pre-initialize crop centered on the displayed image
                if (imgRef.current) {
                  const { width: dw, height: dh } = imgRef.current.getBoundingClientRect();
                  const scaleX = imgRef.current.naturalWidth / dw;
                  const scaleY = imgRef.current.naturalHeight / dh;
                  const pw = Math.min(cropWidth / scaleX, dw);
                  const ph = Math.min(cropHeight / scaleY, dh);
                  setCrop({ unit: "px", x: (dw - pw) / 2, y: (dh - ph) / 2, width: pw, height: ph });
                }
              }}
                className="flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white text-xs rounded-lg transition-colors">
                <Crop className="w-3.5 h-3.5" /> Crop
              </button>
            ) : (
              <>
                <button onClick={() => { setCropMode(false); setCrop(undefined); }}
                  className="flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white text-xs rounded-lg transition-colors">
                  <RotateCcw className="w-3.5 h-3.5" /> Cancel
                </button>
                <button
                  disabled={!crop?.width || saving}
                  onClick={async () => {
                    if (!crop?.width || !imgRef.current || !selected) return;
                    setSaving(true);
                    try {
                      const el = imgRef.current;
                      const scaleX = el.naturalWidth / el.width;
                      const scaleY = el.naturalHeight / el.height;
                      const updated = await cropImage(
                        selected.id,
                        crop.x * scaleX,
                        crop.y * scaleY,
                        crop.width * scaleX,
                        crop.height * scaleY,
                      );
                      // Stay in lightbox — show cropped result immediately
                      setSelected(updated);
                      setImages((prev) => prev.map((img) => img.id === updated.id ? updated : img));
                      setImgBust(Date.now());
                      setCropMode(false);
                      setCrop(undefined);
                    } finally {
                      setSaving(false);
                    }
                  }}
                  className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-xs rounded-lg transition-colors">
                  {saving ? "Saving…" : <><Check className="w-3.5 h-3.5" /> Apply crop</>}
                </button>
              </>
            )}
            <button onClick={() => { setLightbox(false); setCropMode(false); setCrop(undefined); }}
              className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors ml-2">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Image / crop area */}
          <div onClick={(e) => e.stopPropagation()}>
            {cropMode ? (
              <ReactCrop crop={crop} onChange={(c) => setCrop(c)} className="max-h-[80vh]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img ref={imgRef} src={imgSrc(selected, imgBust)} alt=""
                  className="max-w-[90vw] max-h-[80vh] object-contain" />
              </ReactCrop>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img ref={imgRef} src={imgSrc(selected, imgBust)} alt=""
                className="max-w-[92vw] max-h-[85vh] object-contain rounded-lg shadow-2xl" />
            )}
          </div>

          {selected.taxon && !cropMode && (
            <p className="text-white/70 text-sm italic bg-black/50 px-3 py-1 rounded-full">
              {selected.taxon.scientific_name}
            </p>
          )}
        </div>
      )}

      {/* Right: detail panel */}
      {selected && (
        <div className="w-72 shrink-0 bg-[var(--bg-card)] border border-[var(--border)] rounded-xl flex flex-col overflow-hidden">

          {/* Top: scrollable — selected image + metadata + actions */}
          <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-4">
            <div className="relative group aspect-video rounded-lg overflow-hidden bg-[var(--bg-input)] cursor-zoom-in"
                 onClick={() => setLightbox(true)}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={imgSrc(selected, imgBust)} alt="" className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                <Maximize2 className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider">Source</p>
              <p className="text-sm text-[var(--text-base)]">{selected.source_name}</p>
              {selected.taxon && (
                <p className="text-sm text-green-400 italic">{selected.taxon.scientific_name}</p>
              )}
              {selected.author && <p className="text-xs text-[var(--text-dim)]">© {selected.author}</p>}
              {selected.license && (
                <span className="inline-block px-2 py-0.5 text-[10px] bg-[var(--border)] text-[var(--text-dim)] rounded">
                  {selected.license}
                </span>
              )}
              {selected.source_page_url && (
                <a href={selected.source_page_url} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 mt-1">
                  <ExternalLink className="w-3 h-3" /> View source
                </a>
              )}
            </div>

            <div className="space-y-2">
              <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Actions · V/R/D</p>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={() => handleStatus(selected.id, "validated")}
                  className="flex items-center justify-center gap-1.5 py-2 bg-green-600/20 hover:bg-green-600/30 border border-green-500/30 text-green-400 text-xs rounded-lg transition-colors">
                  <Check className="w-3.5 h-3.5" /> Validate
                </button>
                <button onClick={() => handleStatus(selected.id, "rejected")}
                  className="flex items-center justify-center gap-1.5 py-2 bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 text-red-400 text-xs rounded-lg transition-colors">
                  <X className="w-3.5 h-3.5" /> Reject
                </button>
                <button onClick={() => handleStatus(selected.id, "duplicate")}
                  className="flex items-center justify-center gap-1.5 py-2 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 text-purple-400 text-xs rounded-lg transition-colors">
                  <Copy className="w-3.5 h-3.5" /> Duplicate
                </button>
                <button onClick={() => handleStatus(selected.id, "review_later")}
                  className="flex items-center justify-center gap-1.5 py-2 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 text-blue-400 text-xs rounded-lg transition-colors">
                  <Clock className="w-3.5 h-3.5" /> Later
                </button>
              </div>
            </div>
          </div>

          {/* Draggable separator — only shown when there's a taxon */}
          {selected.taxon && (
            <div
              onMouseDown={handleDividerMouseDown}
              className="group h-2.5 flex items-center justify-center cursor-row-resize shrink-0 bg-[var(--bg-input)] hover:bg-amber-500/20 transition-colors border-y border-[var(--border)]"
              title="Drag to resize"
            >
              <div className="w-8 h-0.5 rounded-full bg-[var(--border)] group-hover:bg-amber-400 transition-colors" />
            </div>
          )}

          {/* Bottom: reference image panel (resizable) */}
          {selected.taxon && (
            <div
              style={{ height: refPanelHeight }}
              className="shrink-0 overflow-hidden p-3 flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider flex items-center gap-1">
                  <BookImage className="w-3 h-3" /> Reference — {selected.taxon.scientific_name}
                </p>
                {selected.taxon.reference_image_id === selected.id && (
                  <span className="flex items-center gap-0.5 text-[9px] text-amber-400">
                    <Star className="w-2.5 h-2.5 fill-amber-400" /> Current
                  </span>
                )}
              </div>

              {selected.taxon.reference_image_id ? (
                <div className="relative flex-1 min-h-0 rounded-lg overflow-hidden border border-amber-500/30 bg-[var(--bg-input)]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API_BASE}/images/${selected.taxon.reference_image_id}/thumbnail`}
                    alt="reference"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-1 left-1 flex items-center gap-0.5 px-1.5 py-0.5 bg-amber-500/80 rounded text-[9px] text-white font-medium">
                    <Star className="w-2.5 h-2.5 fill-white" /> Référence
                  </div>
                  {selected.taxon.reference_image_id !== selected.id && (
                    <button
                      disabled={settingRef}
                      onClick={async () => {
                        if (!selected.taxon) return;
                        setSettingRef(true);
                        try {
                          const updated = await setTaxonReference(selected.taxon.id, selected.id);
                          setSelected((prev) => prev ? { ...prev, taxon: updated } : prev);
                          setImages((prev) => prev.map((img) =>
                            img.taxon?.id === updated.id ? { ...img, taxon: updated } : img
                          ));
                        } finally {
                          setSettingRef(false);
                        }
                      }}
                      className="absolute bottom-1 right-1 px-2 py-0.5 bg-black/70 hover:bg-amber-600/80 text-[9px] text-white rounded transition-colors">
                      Changer
                    </button>
                  )}
                </div>
              ) : (
                <button
                  disabled={settingRef}
                  onClick={async () => {
                    if (!selected.taxon) return;
                    setSettingRef(true);
                    try {
                      const updated = await setTaxonReference(selected.taxon.id, selected.id);
                      setSelected((prev) => prev ? { ...prev, taxon: updated } : prev);
                      setImages((prev) => prev.map((img) =>
                        img.taxon?.id === updated.id ? { ...img, taxon: updated } : img
                      ));
                    } finally {
                      setSettingRef(false);
                    }
                  }}
                  className="flex-1 flex items-center justify-center gap-1.5 border border-dashed border-amber-500/40 hover:border-amber-500/70 hover:bg-amber-500/5 text-amber-400 text-xs rounded-lg transition-colors">
                  {settingRef
                    ? <div className="w-3 h-3 border border-amber-400 border-t-transparent rounded-full animate-spin" />
                    : <><Star className="w-3.5 h-3.5" /> Définir comme référence</>
                  }
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
