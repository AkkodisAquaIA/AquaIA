"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Plus, Check, Users, Loader2, Pencil, Trash2, Lock } from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { createUser, deleteUser, updateUser, getUsers } from "@/lib/api";
import type { User } from "@/types";
import { cn } from "@/lib/utils";

export default function WorkspaceSelector() {
  const { currentUserId, currentUserName, workspaces, setWorkspace, setWorkspaces, openLoginModal } = useAppStore();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [saving, setSaving] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setCreating(false);
        setEditingId(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const refreshWorkspaces = async () => {
    const users = await getUsers();
    setWorkspaces(users);
    return users;
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setSaving(true);
    try {
      const user = await createUser(newName.trim());
      await refreshWorkspaces();
      setWorkspace(user.id, user.display_name);
      setNewName("");
      setCreating(false);
    } finally {
      setSaving(false);
    }
  };

  const handleRename = async (id: number) => {
    if (!editName.trim()) return;
    setSaving(true);
    try {
      const updated = await updateUser(id, editName.trim());
      await refreshWorkspaces();
      if (id === currentUserId) setWorkspace(id, updated.display_name);
      setEditingId(null);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (user: User, e: React.MouseEvent) => {
    e.stopPropagation();
    if (workspaces.length <= 1) return;
    await deleteUser(user.id);
    const updated = await refreshWorkspaces();
    if (user.id === currentUserId && updated.length > 0) {
      setWorkspace(updated[0].id, updated[0].display_name);
    }
  };

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-input)] hover:border-[var(--border-hi)] transition-colors text-sm"
      >
        <Users className="w-3.5 h-3.5 text-[var(--text-muted)]" />
        <span className="text-[var(--text-base)] font-medium max-w-[140px] truncate">
          {currentUserName || "…"}
        </span>
        <ChevronDown className={cn("w-3.5 h-3.5 text-[var(--text-muted)] transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-64 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-[var(--border)]">
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Workspaces</p>
          </div>

          <div className="max-h-56 overflow-y-auto divide-y divide-[var(--border)]/40">
            {workspaces.map((ws) => (
              <div key={ws.id} className="group flex items-center gap-2 px-3 py-2 hover:bg-[var(--bg-input)] transition-colors">
                {editingId === ws.id ? (
                  <div className="flex items-center gap-1.5 flex-1 min-w-0">
                    <input
                      autoFocus
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") handleRename(ws.id); if (e.key === "Escape") setEditingId(null); }}
                      className="flex-1 min-w-0 bg-[var(--bg-input)] border border-green-500/40 rounded px-2 py-0.5 text-xs text-[var(--text-base)] focus:outline-none"
                    />
                    <button
                      onClick={() => handleRename(ws.id)}
                      disabled={saving}
                      className="p-1 rounded bg-green-600/20 hover:bg-green-600/40 text-green-400 transition-colors"
                    >
                      {saving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => {
                      if (ws.is_protected && typeof window !== "undefined" && !localStorage.getItem(`adiab-token-${ws.id}`)) {
                        openLoginModal(ws.id, ws.display_name);
                        setOpen(false);
                      } else {
                        setWorkspace(ws.id, ws.display_name);
                        setOpen(false);
                      }
                    }}
                    className="flex-1 min-w-0 flex items-center gap-2 text-left"
                  >
                    {ws.id === currentUserId
                      ? <Check className="w-3.5 h-3.5 text-green-400 shrink-0" />
                      : <span className="w-3.5 h-3.5 shrink-0" />
                    }
                    <span className={cn("text-sm truncate", ws.id === currentUserId ? "text-green-400 font-medium" : "text-[var(--text-base)]")}>
                      {ws.display_name}
                    </span>
                    {ws.is_protected && (
                      <Lock className="w-3 h-3 text-amber-400 shrink-0 ml-auto" />
                    )}
                  </button>
                )}

                {editingId !== ws.id && (
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    <button
                      onClick={(e) => { e.stopPropagation(); setEditingId(ws.id); setEditName(ws.display_name); }}
                      className="p-1 rounded hover:bg-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text-dim)] transition-colors"
                      title="Rename"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                    {workspaces.length > 1 && (
                      <button
                        onClick={(e) => handleDelete(ws, e)}
                        className="p-1 rounded hover:bg-red-500/20 text-[var(--text-muted)] hover:text-red-400 transition-colors"
                        title="Delete workspace"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="border-t border-[var(--border)] p-2">
            {creating ? (
              <div className="flex items-center gap-2">
                <input
                  autoFocus
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); if (e.key === "Escape") setCreating(false); }}
                  placeholder="Workspace name…"
                  className="flex-1 bg-[var(--bg-input)] border border-green-500/40 rounded-lg px-2 py-1.5 text-xs text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none"
                />
                <button
                  onClick={handleCreate}
                  disabled={saving || !newName.trim()}
                  className="p-1.5 rounded-lg bg-green-600/20 hover:bg-green-600/40 text-green-400 disabled:opacity-40 transition-colors"
                >
                  {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                </button>
              </div>
            ) : (
              <button
                onClick={() => setCreating(true)}
                className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-[var(--text-dim)] hover:text-[var(--text-base)] hover:bg-[var(--bg-input)] rounded-lg transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                Add workspace
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
