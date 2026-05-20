import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ADIAB — AquaIA Dataset Builder",
  description: "Professional AI dataset platform for aquatic macro-invertebrate identification",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0a0f] text-white antialiased">{children}</body>
    </html>
  );
}
