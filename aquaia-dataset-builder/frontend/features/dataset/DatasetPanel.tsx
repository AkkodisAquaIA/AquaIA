"use client";

import { useEffect, useState } from "react";
import { getImages, getTaxons } from "@/lib/api";
import type { ImageRecord, Taxon } from "@/types";
import { formatDate } from "@/lib/utils";
import { Database } from "lucide-react";

export default function DatasetPanel() {
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [taxons, setTaxons] = useState<Taxon[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"images" | "taxons">("images");

  useEffect(() => {
    Promise.all([
      getImages({ status: "validated", size: 100 }),
      getTaxons(),
    ]).then(([imgs, txs]) => {
      setImages(imgs.items);
      setTaxons(txs);
    }).finally(() => setLoading(false));
  }, []);

  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Dataset Explorer</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Browse validated images and taxonomy</p>
      </div>

      <div className="flex gap-1.5">
        {(["images", "taxons"] as const).map((tab) => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              activeTab === tab
                ? "bg-green-500/10 border-green-500/30 text-green-400"
                : "bg-[var(--bg-input)] border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--border-hi)]"
            }`}>
            {tab === "images" ? `Validated (${images.length})` : `Taxons (${taxons.length})`}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : activeTab === "images" ? (
        images.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Database className="w-12 h-12 text-[var(--text-ghost)] mb-3" />
            <p className="text-sm text-[var(--text-dim)]">No validated images yet</p>
            <p className="text-xs text-[var(--text-muted)] mt-1">Validate images in the Validation Queue first</p>
          </div>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 xl:grid-cols-8 gap-2">
            {images.map((img) => (
              <div key={img.id} className="aspect-square rounded-lg overflow-hidden border border-[var(--border)] bg-[var(--bg-input)]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={img.source_image_url} alt="" className="w-full h-full object-cover" loading="lazy" />
              </div>
            ))}
          </div>
        )
      ) : (
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
      )}
    </div>
  );
}
