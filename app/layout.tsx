import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ashish T Vasant — Applied AI Engineer",
  description: "Selected work in LLM systems, private local AI, real-time agents and spatial computing.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
