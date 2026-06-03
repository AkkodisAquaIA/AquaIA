"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import axios from "axios";

interface AutocompleteProps {
  value: string;
  onChange: (v: string) => void;
  onSelect: (v: string) => void;
  onEnter?: () => void;
  placeholder?: string;
  className?: string;
}

export default function Autocomplete({
  value,
  onChange,
  onSelect,
  onEnter,
  placeholder,
  className,
}: AutocompleteProps) {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (value.length < 2) { setSuggestions([]); setOpen(false); return; }

    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const { data } = await axios.get<string[]>(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"}/taxonomy/suggest`,
          { params: { q: value } }
        );
        setSuggestions(data);
        setOpen(data.length > 0);
        setActiveIdx(-1);
      } catch {
        setSuggestions([]);
      } finally {
        setLoading(false);
      }
    }, 280);
  }, [value]);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, -1));
    } else if (e.key === "Enter") {
      if (activeIdx >= 0 && suggestions[activeIdx]) {
        onSelect(suggestions[activeIdx]);
        setOpen(false);
      } else {
        onEnter?.();
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div ref={containerRef} className="relative flex-1">
      <div className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className={cn(
            "w-full bg-[var(--bg-input)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-base)]",
            "placeholder-[var(--text-muted)] focus:outline-none focus:border-green-500/50 transition-colors pr-8",
            className
          )}
        />
        {loading && (
          <Loader2 className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--text-muted)] animate-spin" />
        )}
      </div>

      {open && suggestions.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 bg-[var(--bg-input)] border border-[var(--border)] rounded-lg shadow-xl overflow-hidden">
          {suggestions.map((s, i) => (
            <li key={s}>
              <button
                onMouseDown={(e) => { e.preventDefault(); onSelect(s); setOpen(false); }}
                className={cn(
                  "w-full text-left px-3 py-2 text-sm italic transition-colors",
                  i === activeIdx
                    ? "bg-green-500/10 text-green-400"
                    : "text-[var(--text-sub)] hover:bg-[var(--border)] hover:text-[var(--text-base)]"
                )}
              >
                {s}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
