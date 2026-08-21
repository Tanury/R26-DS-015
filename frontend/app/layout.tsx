import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroVoice AI Research System",
  description: "Research neurological risk assessment interface for voice, biomedical marker, and EEG evidence",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
