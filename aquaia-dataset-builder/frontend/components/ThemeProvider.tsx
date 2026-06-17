"use client";

import { useEffect } from "react";
import { useAppStore } from "@/store/appStore";

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { toggleTheme } = useAppStore.getState();

  useEffect(() => {
    const saved = localStorage.getItem("adiab-theme") as "dark" | "light" | null;
    if (saved === "dark") {
      document.documentElement.classList.add("dark");
      useAppStore.setState({ theme: "dark" });
    }
  }, []);

  return <>{children}</>;
}
