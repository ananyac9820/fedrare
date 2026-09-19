import type { Metadata } from "next";
import localFont from "next/font/local";
import { Footer, Header } from "@/components/Chrome";
import { Providers } from "@/components/Providers";
import { dataset } from "@/lib/data";
import "./globals.css";

// Self-hosted variable fonts from @fontsource-variable/* - no network needed to build or run.
const inter = localFont({
  src: "../../node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2",
  variable: "--font-inter",
  weight: "100 900",
  display: "swap",
});
const fraunces = localFont({
  src: "../../node_modules/@fontsource-variable/fraunces/files/fraunces-latin-full-normal.woff2",
  variable: "--font-fraunces",
  weight: "100 900",
  display: "swap",
});
const jetbrains = localFont({
  src: "../../node_modules/@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2",
  variable: "--font-jetbrains",
  weight: "100 800",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "EARN - Earned trust for rare diseases", template: "%s · EARN" },
  description:
    "Research prototype: federated learning across six real hospitals for rare skin-disease " +
    "diagnosis, with a coverage-aware, ledger-anchored trust mechanism (in progress).",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable} ${jetbrains.variable} antialiased`}>
      <body className="min-h-screen font-sans">
        <Providers>
          <Header />
          <main>{children}</main>
          <Footer syncedAt={dataset._meta.syncedAt} />
        </Providers>
      </body>
    </html>
  );
}
