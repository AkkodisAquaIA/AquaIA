"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { Lock, Loader2, X } from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { loginWorkspace } from "@/lib/api";

export default function WorkspaceLoginModal() {
  const { loginModal, closeLoginModal, setWorkspace, setReadOnly, currentUserId } = useAppStore();
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (loginModal) {
      setPassword("");
      setError("");
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [loginModal]);

  if (!loginModal) return null;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!password) return;
    setLoading(true);
    setError("");
    try {
      const data = await loginWorkspace(loginModal.userId, password);
      if (typeof window !== "undefined") {
        localStorage.setItem(`adiab-token-${loginModal.userId}`, data.access_token);
      }
      // If logging into a different workspace (from selector), switch to it
      if (loginModal.userId !== currentUserId) {
        setWorkspace(loginModal.userId, loginModal.displayName, false);
      } else {
        setReadOnly(false);
      }
      closeLoginModal();
    } catch {
      setError("Incorrect password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-6 w-full max-w-sm shadow-2xl mx-4">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
              <Lock className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-[var(--text-base)]">Protected workspace</h2>
              <p className="text-xs text-[var(--text-dim)] mt-0.5">
                <span className="italic">{loginModal.displayName}</span> requires a password
              </p>
            </div>
          </div>
          <button
            onClick={closeLoginModal}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-input)] text-[var(--text-muted)] hover:text-[var(--text-base)] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              ref={inputRef}
              type="password"
              value={password}
              onChange={(e) => { setPassword(e.target.value); setError(""); }}
              placeholder="Password"
              className="w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-xl px-3 py-2.5 text-sm text-[var(--text-base)] placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors"
            />
            {error && (
              <p className="text-xs text-red-400 mt-1.5">{error}</p>
            )}
          </div>

          <button
            type="submit"
            disabled={loading || !password}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
            {loading ? "Verifying…" : "Unlock workspace"}
          </button>
        </form>
      </div>
    </div>
  );
}
