"use client";

export default function SettingsPanel() {
  return (
    <div className="panel-enter space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-white">Settings</h1>
        <p className="text-sm text-[#888899] mt-1">Platform configuration</p>
      </div>

      <div className="bg-[#111118] border border-[#2a2a3a] rounded-xl divide-y divide-[#2a2a3a]">
        {[
          { label: "API URL", value: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api" },
          { label: "Version", value: "0.1.0" },
          { label: "Database", value: "SQLite" },
          { label: "Storage", value: "./storage/" },
        ].map(({ label, value }) => (
          <div key={label} className="flex items-center justify-between px-4 py-3">
            <span className="text-sm text-[#888899]">{label}</span>
            <span className="text-sm text-white font-mono">{value}</span>
          </div>
        ))}
      </div>

      <div className="bg-[#111118] border border-[#2a2a3a] rounded-xl p-4">
        <p className="text-sm font-medium text-white mb-2">Connectors</p>
        <div className="space-y-2">
          {["Wikimedia Commons", "iNaturalist"].map((c) => (
            <div key={c} className="flex items-center justify-between">
              <span className="text-sm text-[#888899]">{c}</span>
              <span className="px-2 py-0.5 text-[10px] bg-green-500/10 text-green-400 border border-green-500/20 rounded">
                active
              </span>
            </div>
          ))}
          {["GBIF"].map((c) => (
            <div key={c} className="flex items-center justify-between">
              <span className="text-sm text-[#888899]">{c}</span>
              <span className="px-2 py-0.5 text-[10px] bg-[#2a2a3a] text-[#555566] border border-[#3a3a50] rounded">
                coming soon
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
