"use client";

import { useEffect } from "react";
import { useAppStore } from "@/store/appStore";

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { toggleTheme } = useAppStore.getState();

  useEffect(() => {
    const saved = localStorage.getItem("adiab-theme") as "dark" | "light" | null;
    if (saved === "light") {
      document.documentElement.classList.add("light");
      useAppStore.setState({ theme: "light" });
    }
  }, []);

  return <>{children}</>;
}
