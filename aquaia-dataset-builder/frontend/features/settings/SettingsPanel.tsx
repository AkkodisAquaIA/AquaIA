"use client";

import { useState, useEffect } from "react";
import { useAppStore } from "@/store/appStore";
import { cn } from "@/lib/utils";
import { Star, X } from "lucide-react";
import { getTaxons, setTaxonReference } from "@/lib/api";
import type { Taxon } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

const CROP_PRESETS = [
  { label: "224 × 224", w: 224, h: 224 },
  { label: "256 × 256", w: 256, h: 256 },
  { label: "320 × 320", w: 320, h: 320 },
  { label: "416 × 416", w: 416, h: 416 },
  { label: "512 × 512", w: 512, h: 512 },
  { label: "640 × 640", w: 640, h: 640 },
];

export default function SettingsPanel() {
  const { cropWidth, cropHeight, setCropDimensions, currentUserId } = useAppStore();
  const [customW, setCustomW] = useState(String(cropWidth));
  const [customH, setCustomH] = useState(String(cropHeight));
  const [taxons, setTaxons] = useState<Taxon[]>([]);

  useEffect(() => {
    if (!currentUserId) return;
    getTaxons(currentUserId).then(setTaxons).catch(() => {});
  }, [currentUserId]);

  const handleClearReference = async (taxon: Taxon) => {
    if (!currentUserId) return;
    const updated = await setTaxonReference(taxon.id, null, currentUserId);
    setTaxons((prev) => prev.map((t) => (t.id === updated.id ? updated : t)));
  };

  const applyCustom = () => {
    const w = parseInt(customW);
    const h = parseInt(customH);
    if (w > 0 && h > 0) setCropDimensions(w, h);
  };

  const isPreset = CROP_PRESETS.some((p) => p.w === cropWidth && p.h === cropHeight);

  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Settings</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Platform configuration</p>
      </div>

      {/* System info */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl divide-y divide-[var(--border)]">
        {[
          { label: "API URL",   value: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api" },
          { label: "Version",   value: "0.2.0" },
          { label: "Database",  value: "SQLite" },
          { label: "Storage",   value: "./storage/" },
        ].map(({ label, value }) => (
          <div key={label} className="flex items-center justify-between px-4 py-3">
            <span className="text-sm text-[var(--text-dim)]">{label}</span>
            <span className="text-sm text-[var(--text-base)] font-mono">{value}</span>
          </div>
        ))}
      </div>

      {/* Default crop dimensions */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-4">
        <div>
          <p className="text-sm font-medium text-[var(--text-base)]">Default crop dimensions</p>
          <p className="text-xs text-[var(--text-dim)] mt-0.5">
            When you open the crop tool, the selection is pre-sized to these dimensions and centered on the image.
            Current: <span className="font-mono text-green-400">{cropWidth} × {cropHeight} px</span>
          </p>
        </div>

        {/* Presets */}
        <div className="flex flex-wrap gap-2">
          {CROP_PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => { setCropDimensions(p.w, p.h); setCustomW(String(p.w)); setCustomH(String(p.h)); }}
              className={cn(
                "px-3 py-1.5 text-xs rounded-lg border transition-colors font-mono",
                cropWidth === p.w && cropHeight === p.h
                  ? "bg-green-500/10 border-green-500/30 text-green-400"
                  : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
              )}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Custom */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--text-muted)] shrink-0">Custom:</span>
          <input
            type="number" min={32} max={4096} value={customW}
            onChange={(e) => setCustomW(e.target.value)}
            onBlur={applyCustom}
            onKeyDown={(e) => e.key === "Enter" && applyCustom()}
            className="w-20 bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-center font-mono text-[var(--text-base)] focus:outline-none focus:border-green-500/50"
          />
          <span className="text-xs text-[var(--text-muted)]">×</span>
          <input
            type="number" min={32} max={4096} value={customH}
            onChange={(e) => setCustomH(e.target.value)}
            onBlur={applyCustom}
            onKeyDown={(e) => e.key === "Enter" && applyCustom()}
            className="w-20 bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-center font-mono text-[var(--text-base)] focus:outline-none focus:border-green-500/50"
          />
          <span className="text-xs text-[var(--text-muted)]">px</span>
          {!isPreset && (
            <span className="px-2 py-0.5 text-[10px] bg-green-500/10 text-green-400 border border-green-500/20 rounded font-mono ml-1">
              {cropWidth} × {cropHeight}
            </span>
          )}
        </div>
      </div>

      {/* Reference images */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
        <div>
          <p className="text-sm font-medium text-[var(--text-base)]">Reference images</p>
          <p className="text-xs text-[var(--text-dim)] mt-0.5">
            One representative image per species, shown in the Validation Queue for comparison. Set from the detail panel while reviewing.
          </p>
        </div>
        {taxons.filter((t) => t.reference_image_id).length === 0 ? (
          <p className="text-xs text-[var(--text-muted)] italic">
            No reference images defined yet. Select an image in the Validation Queue and click "Définir comme référence".
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-3">
            {taxons.filter((t) => t.reference_image_id).map((t) => (
              <div key={t.id} className="group relative rounded-lg overflow-hidden border border-amber-500/30 bg-[var(--bg-input)]">
                <div className="aspect-square">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API_BASE}/images/${t.reference_image_id}/thumbnail`}
                    alt={t.scientific_name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="px-2 py-1.5 space-y-0.5">
                  <p className="text-[10px] text-[var(--text-base)] italic truncate font-medium">{t.scientific_name}</p>
                  {t.common_name && (
                    <p className="text-[9px] text-[var(--text-muted)] truncate">{t.common_name}</p>
                  )}
                </div>
                <div className="absolute top-1 left-1 flex items-center gap-0.5 px-1 py-0.5 bg-amber-500/80 rounded text-[9px] text-white">
                  <Star className="w-2.5 h-2.5 fill-white" />
                </div>
                <button
                  onClick={() => handleClearReference(t)}
                  className="absolute top-1 right-1 p-1 bg-black/60 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/80"
                  title="Remove reference">
                  <X className="w-3 h-3 text-white" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Connectors */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4">
        <p className="text-sm font-medium text-[var(--text-base)] mb-2">Connectors</p>
        <div className="space-y-2">
          {["Wikimedia Commons", "iNaturalist", "GBIF"].map((c) => (
            <div key={c} className="flex items-center justify-between">
              <span className="text-sm text-[var(--text-dim)]">{c}</span>
              <span className="px-2 py-0.5 text-[10px] bg-green-500/10 text-green-400 border border-green-500/20 rounded">
                active
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
