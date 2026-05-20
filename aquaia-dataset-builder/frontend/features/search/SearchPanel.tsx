"use client";

import { useState } from "react";
import { Search, Loader2, ExternalLink, CheckCircle } from "lucide-react";
import { runSearch } from "@/lib/api";
import { useAppStore } from "@/store/appStore";
import type { ImageRecord } from "@/types";
import { cn } from "@/lib/utils";
import Autocomplete from "@/components/ui/Autocomplete";

const SOURCES = [
  { id: "wikimedia", label: "Wikimedia" },
  { id: "inaturalist", label: "iNaturalist" },
];

export default function SearchPanel() {
  const { searchQuery, setSearchQuery, selectedSources, toggleSource } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ImageRecord[]>([]);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runSearch(searchQuery.trim(), selectedSources, 50);
      setResults(data);
      setSearched(true);
    } catch {
      setError("Search failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Image Search</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Search macro-invertebrate images from biodiversity platforms</p>
      </div>

      {/* Search bar */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
        <div className="flex gap-2">
          <Autocomplete
            value={searchQuery}
            onChange={setSearchQuery}
            onSelect={(v) => { setSearchQuery(v); }}
            onEnter={handleSearch}
            placeholder="e.g. Ephemeroptera, Baetis rhodani..."
          />
          <button
            onClick={handleSearch}
            disabled={loading || !searchQuery.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        {/* Source toggles */}
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
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Results grid */}
      {searched && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-[var(--text-dim)]">
              {results.length > 0
                ? `${results.length} new images retrieved`
                : "No new images found (already in database or no results)"}
            </p>
          </div>

          {results.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-2">
              {results.map((img) => (
                <ImageCard key={img.id} image={img} />
              ))}
            </div>
          )}
        </div>
      )}

      {!searched && (
        <div className="flex flex-col items-center justify-center h-48 text-center">
          <Search className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
          <p className="text-sm text-[var(--text-dim)]">Enter a scientific name to start searching</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">e.g. Ephemeroptera, Plecoptera, Trichoptera</p>
        </div>
      )}
    </div>
  );
}

function ImageCard({ image }: { image: ImageRecord }) {
  return (
    <div className="group relative bg-[var(--bg-card)] border border-[var(--border)] rounded-lg overflow-hidden hover:border-green-500/30 transition-colors">
      <div className="aspect-square bg-[var(--bg-input)] relative overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={image.source_image_url}
          alt={image.taxon?.scientific_name ?? "image"}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
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
