"use client";

import Link from "next/link";
import { motion, useScroll, useSpring } from "framer-motion";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { REPO } from "@/lib/data";
import { NAV, isActive } from "@/lib/nav";

const LINKS = NAV.map((n) => [n.href, n.label] as const);

/** Thin bar showing how far through the page you are - the site reads as one path. */
function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const width = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.3 });
  return (
    <motion.div
      aria-hidden
      className="absolute inset-x-5 bottom-0 h-px origin-left bg-accent md:inset-x-7"
      style={{ scaleX: width }}
    />
  );
}

export function Header() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed inset-x-0 top-0 z-50 px-4 pt-4 md:px-6">
      <nav
        aria-label="Main"
        className="mx-auto flex h-16 max-w-6xl items-center gap-6 rounded-2xl border border-line bg-paper/90 px-5 shadow-[0_8px_30px_-20px_rgba(29,41,41,0.35)] backdrop-blur-md md:px-7"
      >
        <Link href="/" className="font-display text-2xl font-semibold tracking-tight text-ink" onClick={() => setOpen(false)}>
          earn<span className="text-accent">.</span>
        </Link>
        <ul className="ml-auto hidden items-center gap-7 lg:flex">
          {LINKS.map(([href, label]) => {
            const active = isActive(pathname, href);
            return (
              <li key={href}>
                <Link
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={`font-mono text-[11px] uppercase tracking-[0.16em] transition ${
                    active ? "text-ink underline decoration-accent decoration-2 underline-offset-8" : "text-muted hover:text-ink"
                  }`}
                >
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
        <Link
          href="/demo"
          className="hidden items-center gap-2 rounded-full bg-accent px-4 py-2 text-xs font-semibold text-cream transition hover:bg-ink lg:inline-flex"
        >
          <span aria-hidden className="block h-1.5 w-1.5 rounded-full bg-cream/80" />
          Run demo
        </Link>
        <a
          href={REPO}
          target="_blank"
          rel="noreferrer"
          className="hidden rounded-full border border-ink/20 px-4 py-2 text-xs font-semibold text-ink transition hover:border-ink/50 lg:inline-flex"
        >
          GitHub ↗
        </a>
        <ScrollProgress />
        <button
          type="button"
          className="ml-auto rounded-full border border-line px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-ink lg:hidden"
          aria-expanded={open}
          aria-controls="mobile-nav"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Close" : "Menu"}
        </button>
      </nav>
      {open && (
        <div id="mobile-nav" className="mx-auto mt-2 max-w-6xl rounded-2xl border border-line bg-paper p-3 shadow-lg lg:hidden">
          <ul className="grid gap-1">
            {LINKS.map(([href, label]) => {
              const active = isActive(pathname, href);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setOpen(false)}
                    className={`block rounded-xl px-4 py-3 font-mono text-xs uppercase tracking-[0.16em] ${
                      active ? "bg-accent-soft text-accent" : "text-muted hover:bg-sand"
                    }`}
                  >
                    {label}
                  </Link>
                </li>
              );
            })}
            <li>
              <Link
                href="/demo"
                onClick={() => setOpen(false)}
                className="block rounded-xl bg-accent px-4 py-3 font-mono text-xs uppercase tracking-[0.16em] text-cream"
              >
                Run demo
              </Link>
            </li>
            <li>
              <a href={REPO} target="_blank" rel="noreferrer" className="block rounded-xl px-4 py-3 font-mono text-xs uppercase tracking-[0.16em] text-muted hover:bg-sand">
                GitHub ↗
              </a>
            </li>
          </ul>
        </div>
      )}
    </header>
  );
}

export function Footer({ syncedAt }: { syncedAt: string }) {
  return (
    <footer className="border-t border-line bg-sand/70">
      <div className="mx-auto max-w-6xl px-6 py-24 md:py-32">
        <div className="grid gap-16 md:grid-cols-[1.3fr_1fr]">
          <div>
            <p className="font-display text-4xl font-medium leading-tight tracking-tight text-ink md:text-5xl">
              Earned trust for
              <br />
              rare diseases.
            </p>
            <p className="mt-6 max-w-md text-pretty leading-relaxed text-muted">
              A student research prototype. It shows real measurements where they exist, labels
              everything else, and reports every check exactly as it came out. Not a medical product; no
              clinical claims.
            </p>
          </div>
          <div className="space-y-8">
            <ul className="grid grid-cols-2 gap-y-3 font-mono text-xs uppercase tracking-[0.14em] text-muted">
              {LINKS.map(([href, label]) => (
                <li key={href}>
                  <Link href={href} className="hover:text-ink">
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="mt-20 flex flex-wrap items-center justify-between gap-4 border-t border-sand-deep pt-8 font-mono text-[11px] text-faint">
          <span>Data last synced from results/ on {syncedAt}</span>
          <a href={REPO} className="hover:text-ink" target="_blank" rel="noreferrer">
            {REPO.replace("https://", "")}
          </a>
        </div>
      </div>
    </footer>
  );
}
