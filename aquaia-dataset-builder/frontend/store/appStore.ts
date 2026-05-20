import { create } from "zustand";
import type { PanelId, ImageRecord } from "@/types";

interface AppState {
  theme: "dark" | "light";
  toggleTheme: () => void;

  activePanel: PanelId;
  setActivePanel: (panel: PanelId) => void;

  searchQuery: string;
  setSearchQuery: (q: string) => void;

  selectedSources: string[];
  toggleSource: (source: string) => void;

  selectedImage: ImageRecord | null;
  setSelectedImage: (img: ImageRecord | null) => void;

  validationFilter: string;
  setValidationFilter: (f: string) => void;

  cropWidth: number;
  cropHeight: number;
  setCropDimensions: (w: number, h: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  theme: "light",
  toggleTheme: () =>
    set((s) => {
      const next = s.theme === "dark" ? "light" : "dark";
      if (typeof window !== "undefined") {
        document.documentElement.classList.toggle("dark", next === "dark");
        localStorage.setItem("adiab-theme", next);
      }
      return { theme: next };
    }),

  activePanel: "dashboard",
  setActivePanel: (panel) => set({ activePanel: panel }),

  searchQuery: "",
  setSearchQuery: (q) => set({ searchQuery: q }),

  selectedSources: ["wikimedia", "inaturalist", "gbif"],
  toggleSource: (source) =>
    set((s) => ({
      selectedSources: s.selectedSources.includes(source)
        ? s.selectedSources.filter((x) => x !== source)
        : [...s.selectedSources, source],
    })),

  selectedImage: null,
  setSelectedImage: (img) => set({ selectedImage: img }),

  validationFilter: "pending",
  setValidationFilter: (f) => set({ validationFilter: f }),

  cropWidth: parseInt(typeof window !== "undefined" ? localStorage.getItem("adiab-crop-w") || "640" : "640"),
  cropHeight: parseInt(typeof window !== "undefined" ? localStorage.getItem("adiab-crop-h") || "640" : "640"),
  setCropDimensions: (w, h) => {
    if (typeof window !== "undefined") {
      localStorage.setItem("adiab-crop-w", String(w));
      localStorage.setItem("adiab-crop-h", String(h));
    }
    set({ cropWidth: w, cropHeight: h });
  },
}));
