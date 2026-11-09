import type { Metadata } from "next";
import "./globals.css";
import ThemeProvider from "@/components/ThemeProvider";
import StorageWarning from "@/components/StorageWarning";

export const metadata: Metadata = {
  title: "ADIAB — AquaIA Dataset Builder",
  description: "Professional AI dataset platform for aquatic macro-invertebrate identification",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <StorageWarning />
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
