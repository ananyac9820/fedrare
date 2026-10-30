import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Console",
  description: "The live federated system: hospitals contribute, the model learns, clinics diagnose.",
};

/** The console opts out of the research site's chrome and light theme entirely. */
export default function ConsoleLayout({ children }: { children: React.ReactNode }) {
  return <div className="min-h-dvh bg-[#06060a]">{children}</div>;
}
