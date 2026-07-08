"use client";

import { useCallback, useRef, useState } from "react";
import { Search, Loader2, ExternalLink, CheckCircle, Link2, Upload, X, ImageIcon } from "lucide-react";
import { runSearch, importFromUrl, uploadFiles } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ImageRecord } from "@/types";
import { cn } from "@/lib/utils";
import Autocomplete from "@/components/ui/Autocomplete";

type Mode = "search" | "url" | "upload";

const SOURCES = [
  { id: "wikimedia",   label: "Wikimedia" },
  { id: "inaturalist", label: "iNaturalist" },
  { id: "gbif",        label: "GBIF" },
];

const MODES: { id: Mode; label: string; icon: React.ElementType }[] = [
  { id: "search", label: "Search by name", icon: Search },
  { id: "url",    label: "Add by URL",     icon: Link2 },
  { id: "upload", label: "Upload files",   icon: Upload },
];

export default function SearchPanel() {
  const [mode, setMode] = useState<Mode>("search");

  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Image Search</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Search or import macro-invertebrate images</p>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-1.5">
        {MODES.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setMode(id)}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border transition-colors",
              mode === id
                ? "bg-green-500/10 border-green-500/30 text-green-500"
                : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </div>

      {mode === "search" && <SearchByNameMode />}
      {mode === "url"    && <AddByUrlMode />}
      {mode === "upload" && <UploadMode />}
    </div>
  );
}


// ─── Mode 1: Search by scientific name ──────────────────────────────────────

const LIMIT_OPTIONS = [10, 25, 50, 100, 200];

function SearchByNameMode() {
  const { selectedSources, toggleSource, currentUserId } = useAppStore();
  const [genus, setGenus] = useState("");
  const [species, setSpecies] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ImageRecord[]>([]);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(50);

  const effectiveQuery = [genus.trim(), species.trim()].filter(Boolean).join(" ");

  const handleSearch = async () => {
    if (!effectiveQuery || !currentUserId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runSearch(currentUserId, effectiveQuery, selectedSources, limit);
      setResults(data);
      setSearched(true);
    } catch {
      setError("Search failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
        <div className="flex gap-2">
          <div className="flex flex-1 gap-2">
            <div className="flex-1">
              <label className="text-[10px] text-[var(--text-muted)] mb-1 block">Genus</label>
              <Autocomplete
                value={genus}
                onChange={setGenus}
                onSelect={(v) => {
                  const parts = v.trim().split(/\s+/);
                  if (parts.length >= 2) {
                    setGenus(parts[0]);
                    setSpecies(parts.slice(1).join(" "));
                  } else {
                    setGenus(v);
                  }
                }}
                onEnter={handleSearch}
                placeholder="e.g. Paracorixa"
              />
            </div>
            <div className="flex-1">
              <label className="text-[10px] text-[var(--text-muted)] mb-1 block">Species <span className="text-[var(--text-ghost)]">(optional)</span></label>
              <input
                type="text"
                value={species}
                onChange={(e) => setSpecies(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="e.g. concinna"
                className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors"
              />
            </div>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleSearch}
              disabled={loading || !effectiveQuery}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              {loading ? "Searching…" : "Search"}
            </button>
          </div>
        </div>
        {effectiveQuery && (
          <p className="text-[10px] text-[var(--text-muted)]">
            Query: <span className="italic text-[var(--text-dim)]">{effectiveQuery}</span>
          </p>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[var(--text-muted)]">Sources:</span>
            {SOURCES.map((s) => (
              <button
                key={s.id}
                onClick={() => toggleSource(s.id)}
                className={cn(
                  "px-2.5 py-1 text-xs rounded-md border transition-colors",
                  selectedSources.includes(s.id)
                    ? "bg-green-500/10 border-green-500/30 text-green-400"
                    : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
                )}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-[var(--text-muted)]">Max results:</span>
            <div className="flex gap-1">
              {LIMIT_OPTIONS.map((n) => (
                <button
                  key={n}
                  onClick={() => setLimit(n)}
                  className={cn(
                    "px-2 py-1 text-xs rounded-md border transition-colors",
                    limit === n
                      ? "bg-green-500/10 border-green-500/30 text-green-400"
                      : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
                  )}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {searched && (
        <div>
          <p className="text-sm text-[var(--text-dim)] mb-3">
            {results.length > 0
              ? `${results.length} new images retrieved`
              : "No new images found (already in database or no results)"}
          </p>
          {results.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-2">
              {results.map((img) => <ImageCard key={img.id} image={img} />)}
            </div>
          )}
        </div>
      )}

      {!searched && (
        <div className="flex flex-col items-center justify-center h-48 text-center">
          <Search className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
          <p className="text-sm text-[var(--text-dim)]">Enter a genus or a genus + species to search</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">e.g. Genus: Paracorixa · Species: concinna</p>
        </div>
      )}
    </div>
  );
}


// ─── Mode 2: Add by URL ──────────────────────────────────────────────────────

function AddByUrlMode() {
  const { currentUserId } = useAppStore();
  const [url, setUrl] = useState("");
  const [scientificName, setScientificName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ImageRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAdd = async () => {
    if (!url.trim() || !currentUserId) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const img = await importFromUrl(currentUserId, url.trim(), scientificName.trim() || undefined);
      setResult(img);
      setUrl("");
      setScientificName("");
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg || "Failed to import URL. Make sure it points to a valid image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
        <div>
          <label className="text-xs text-[var(--text-dim)] mb-1 block">Image URL</label>
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            placeholder="https://upload.wikimedia.org/…/Baetis_rhodani.jpg"
            className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors"
          />
        </div>

        <div>
          <label className="text-xs text-[var(--text-dim)] mb-1 block">Scientific name (optional)</label>
          <Autocomplete
            value={scientificName}
            onChange={setScientificName}
            onSelect={setScientificName}
            placeholder="e.g. Baetis rhodani"
          />
        </div>

        <button
          onClick={handleAdd}
          disabled={loading || !url.trim()}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Link2 className="w-4 h-4" />}
          {loading ? "Importing…" : "Add image"}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-green-500/5 border border-green-500/20 rounded-xl p-4 flex items-center gap-4">
          <CheckCircle className="w-5 h-5 text-green-400 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-[var(--text-base)] font-medium">Image added successfully</p>
            <p className="text-xs text-[var(--text-dim)] mt-0.5 truncate">
              {result.taxon?.scientific_name ?? "No taxon"} · ID #{result.id} · downloading in background…
            </p>
          </div>
        </div>
      )}

      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4">
        <p className="text-xs font-medium text-[var(--text-dim)] mb-2">Accepted URLs</p>
        <ul className="space-y-1 text-xs text-[var(--text-muted)]">
          <li>• Direct image link: <span className="font-mono">.jpg .png .webp .gif</span></li>
          <li>• Wikimedia Commons direct file URL</li>
          <li>• iNaturalist photo URL</li>
          <li>• Any publicly accessible image URL</li>
        </ul>
      </div>
    </div>
  );
}


// ─── Mode 3: Upload from disk ────────────────────────────────────────────────

function UploadMode() {
  const { currentUserId } = useAppStore();
  const [files, setFiles] = useState<File[]>([]);
  const [scientificName, setScientificName] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ImageRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const imgs = Array.from(incoming).filter((f) => f.type.startsWith("image/"));
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name + f.size));
      return [...prev, ...imgs.filter((f) => !existing.has(f.name + f.size))];
    });
  };

  const removeFile = (idx: number) =>
    setFiles((prev) => prev.filter((_, i) => i !== idx));

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  }, []);

  const handleUpload = async () => {
    if (!files.length || !currentUserId) return;
    setLoading(true);
    setError(null);
    try {
      const imported = await uploadFiles(currentUserId, files, scientificName.trim() || undefined);
      setResults(imported);
      setFiles([]);
      setScientificName("");
    } catch {
      setError("Upload failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "bg-[var(--bg-card)] border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors",
          dragging
            ? "border-green-500/50 bg-green-500/5"
            : "border-[var(--border)] hover:border-[var(--border-hi)]"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => addFiles(e.target.files)}
        />
        <Upload className="w-8 h-8 text-[var(--text-ghost)] mx-auto mb-3" />
        <p className="text-sm text-[var(--text-dim)]">Drop images here or click to browse</p>
        <p className="text-xs text-[var(--text-muted)] mt-1">JPG, PNG, WebP, GIF — multiple files supported</p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
          <div className="px-4 py-2 border-b border-[var(--border)] flex items-center justify-between">
            <span className="text-xs text-[var(--text-dim)]">{files.length} file{files.length > 1 ? "s" : ""} selected</span>
            <button onClick={() => setFiles([])} className="text-xs text-[var(--text-muted)] hover:text-red-400 transition-colors">
              Clear all
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto divide-y divide-[var(--border)]/50">
            {files.map((f, i) => (
              <div key={i} className="flex items-center gap-3 px-4 py-2">
                <ImageIcon className="w-4 h-4 text-[var(--text-muted)] shrink-0" />
                <span className="text-xs text-[var(--text-dim)] flex-1 truncate">{f.name}</span>
                <span className="text-xs text-[var(--text-muted)] shrink-0">
                  {(f.size / 1024).toFixed(0)} KB
                </span>
                <button onClick={() => removeFile(i)} className="text-[var(--text-muted)] hover:text-red-400 transition-colors">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Taxon + submit */}
      {files.length > 0 && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
          <div>
            <label className="text-xs text-[var(--text-dim)] mb-1 block">Scientific name (optional)</label>
            <Autocomplete
              value={scientificName}
              onChange={setScientificName}
              onSelect={setScientificName}
              placeholder="e.g. Baetis rhodani"
            />
          </div>
          <button
            onClick={handleUpload}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            {loading ? "Uploading…" : `Upload ${files.length} file${files.length > 1 ? "s" : ""}`}
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3">
          <div className="bg-green-500/5 border border-green-500/20 rounded-xl p-4 flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-green-400 shrink-0" />
            <p className="text-sm text-[var(--text-base)]">
              {results.length} image{results.length > 1 ? "s" : ""} imported successfully
            </p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-2">
            {results.map((img) => <ImageCard key={img.id} image={img} />)}
          </div>
        </div>
      )}

      {!files.length && !results.length && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4">
          <p className="text-xs font-medium text-[var(--text-dim)] mb-2">How it works</p>
          <ul className="space-y-1 text-xs text-[var(--text-muted)]">
            <li>• Drag & drop or select multiple images from your computer</li>
            <li>• SHA256 + perceptual hash deduplication applied automatically</li>
            <li>• Images are saved to local storage and immediately available for validation</li>
          </ul>
        </div>
      )}
    </div>
  );
}


// ─── Shared image card ───────────────────────────────────────────────────────

function ImageCard({ image }: { image: ImageRecord }) {
  const src = image.local_path
    ? `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"}/images/${image.id}/thumbnail`
    : image.source_image_url;

  return (
    <div className="group relative bg-[var(--bg-card)] border border-[var(--border)] rounded-lg overflow-hidden hover:border-green-500/30 transition-colors">
      <div className="aspect-square bg-[var(--bg-input)] relative overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={image.taxon?.scientific_name ?? "image"}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <a
            href={image.source_page_url ?? image.source_image_url}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1.5 bg-white/10 rounded-lg hover:bg-white/20"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-3.5 h-3.5 text-white" />
          </a>
        </div>
      </div>
      <div className="px-2 py-1.5">
        <p className="text-[10px] text-[var(--text-dim)] truncate">{image.source_name}</p>
        <div className="flex items-center gap-1 mt-0.5">
          <CheckCircle className="w-3 h-3 text-green-400 shrink-0" />
          <span className="text-[10px] text-green-400">saved</span>
        </div>
      </div>
    </div>
  );
}
