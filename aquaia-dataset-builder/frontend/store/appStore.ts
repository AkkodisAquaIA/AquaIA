import { create } from "zustand";
import type { PanelId, ImageRecord, User } from "@/types";
import { getUsers, createUser } from "@/lib/api";

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

  // Workspace
  currentUserId: number | null;
  currentUserName: string;
  workspaces: User[];
  workspaceReady: boolean;
  initWorkspace: () => Promise<void>;
  setWorkspace: (id: number, name: string) => void;
  setWorkspaces: (workspaces: User[]) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
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

  // Workspace
  currentUserId: null,
  currentUserName: "",
  workspaces: [],
  workspaceReady: false,

  initWorkspace: async () => {
    try {
      const users = await getUsers();
      set({ workspaces: users });

      if (users.length === 0) {
        set({ workspaceReady: true });
        return;
      }

      const savedId = typeof window !== "undefined"
        ? parseInt(localStorage.getItem("adiab-workspace-id") || "0")
        : 0;

      const saved = users.find((u) => u.id === savedId);
      const target = saved ?? users[0];

      if (typeof window !== "undefined") {
        localStorage.setItem("adiab-workspace-id", String(target.id));
      }
      set({ currentUserId: target.id, currentUserName: target.display_name, workspaceReady: true });
    } catch {
      set({ workspaceReady: true });
    }
  },

  setWorkspace: (id, name) => {
    if (typeof window !== "undefined") {
      localStorage.setItem("adiab-workspace-id", String(id));
    }
    set({ currentUserId: id, currentUserName: name });
  },

  setWorkspaces: (workspaces) => set({ workspaces }),
}));
