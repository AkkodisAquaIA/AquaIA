"use client";

export default function SettingsPanel() {
  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-base)]">Settings</h1>
        <p className="text-sm text-[var(--text-dim)] mt-1">Platform configuration</p>
      </div>

      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl divide-y divide-[var(--border)]">
        {[
          { label: "API URL", value: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api" },
          { label: "Version", value: "0.1.0" },
          { label: "Database", value: "SQLite" },
          { label: "Storage", value: "./storage/" },
        ].map(({ label, value }) => (
          <div key={label} className="flex items-center justify-between px-4 py-3">
            <span className="text-sm text-[var(--text-dim)]">{label}</span>
            <span className="text-sm text-[var(--text-base)] font-mono">{value}</span>
          </div>
        ))}
      </div>

      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4">
        <p className="text-sm font-medium text-[var(--text-base)] mb-2">Connectors</p>
        <div className="space-y-2">
          {["Wikimedia Commons", "iNaturalist"].map((c) => (
            <div key={c} className="flex items-center justify-between">
              <span className="text-sm text-[var(--text-dim)]">{c}</span>
              <span className="px-2 py-0.5 text-[10px] bg-green-500/10 text-green-400 border border-green-500/20 rounded">
                active
              </span>
            </div>
          ))}
          {["GBIF"].map((c) => (
            <div key={c} className="flex items-center justify-between">
              <span className="text-sm text-[var(--text-dim)]">{c}</span>
              <span className="px-2 py-0.5 text-[10px] bg-[var(--border)] text-[var(--text-muted)] border border-[var(--border-hi)] rounded">
                coming soon
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
