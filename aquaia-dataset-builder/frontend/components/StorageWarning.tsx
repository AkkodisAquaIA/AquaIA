"use client";

import { useEffect, useState } from "react";

interface StorageStatus {
  used_percent: number;
  warning: boolean;
  critical: boolean;
}

export default function StorageWarning() {
  const [status, setStatus] = useState<StorageStatus | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/system/storage`
        );

        const data = await response.json();
        setStatus(data);
      } catch (e) {
        console.error(e);
      }
    };

    load();

    const interval = setInterval(load, 60000);

    return () => clearInterval(interval);
  }, []);

  if (!status?.warning) {
    return null;
  }

  return (
    <div
      className={`w-full py-2 text-center text-white font-semibold ${
        status.critical ? "bg-red-600" : "bg-orange-500"
      }`}
    >
      {status.critical
        ? `⚠️ Stockage critique (${status.used_percent}% utilisé)`
        : `⚠️ Stockage bientôt saturé (${status.used_percent}% utilisé)`}
    </div>
  );
}