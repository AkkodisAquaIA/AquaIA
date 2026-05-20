"use client";

import { useEffect, useState, useCallback } from "react";
import { Check, X, Copy, Clock, ExternalLink, ChevronLeft, ChevronRight } from "lucide-react";
import { getImages, updateImageStatus } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ImageRecord } from "@/types";
import { cn } from "@/lib/utils";

const FILTERS = [
  { id: "pending", label: "Pending" },
  { id: "validated", label: "Validated" },
  { id: "rejected", label: "Rejected" },
  { id: "duplicate", label: "Duplicates" },
];

export default function ValidationPanel() {
  const { validationFilter, setValidationFilter } = useAppStore();
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<ImageRecord | null>(null);

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
    setImages((prev) => prev.filter((img) => img.id !== id));
    setTotal((t) => t - 1);
    setSelected(null);
  };

  // Keyboard shortcuts
  useEffect(() => {
    if (!selected) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "v" || e.key === "V") handleStatus(selected.id, "validated");
      if (e.key === "r" || e.key === "R") handleStatus(selected.id, "rejected");
      if (e.key === "d" || e.key === "D") handleStatus(selected.id, "duplicate");
      if (e.key === " ") { e.preventDefault(); setSelected(null); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

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

      {/* Right: detail panel */}
      {selected && (
        <div className="w-72 shrink-0 bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 flex flex-col gap-4 overflow-y-auto">
          <div className="aspect-video rounded-lg overflow-hidden bg-[var(--bg-input)]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={selected.source_image_url} alt="" className="w-full h-full object-cover" />
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

          {/* Actions */}
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
      )}
    </div>
  );
}
