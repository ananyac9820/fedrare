import Link from "next/link";
import { NAV, navIndex } from "@/lib/nav";

/** Previous / next links at the foot of every page, so the site reads start to end. */
export function Pager({ current }: { current: string }) {
  const i = navIndex(current);
  const prev = i > 0 ? NAV[i - 1] : null;
  const next = i >= 0 && i < NAV.length - 1 ? NAV[i + 1] : null;
  if (!prev && !next) return null;

  return (
    <nav aria-label="Page order" className="border-t border-line bg-sand/40">
      <div className="mx-auto grid max-w-6xl gap-4 px-6 py-14 md:grid-cols-2 md:py-16">
        {prev ? (
          <Link
            href={prev.href}
            className="group rounded-[24px] border border-line bg-paper p-7 transition hover:border-ink/30"
          >
            <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
              &larr; Previous · step {i} of {NAV.length - 1}
            </span>
            <p className="mt-3 font-display text-2xl font-medium text-ink group-hover:text-accent">{prev.label}</p>
            <p className="mt-1 text-sm text-muted">{prev.blurb}</p>
          </Link>
        ) : (
          <span />
        )}
        {next && (
          <Link
            href={next.href}
            className="group rounded-[24px] border border-line bg-paper p-7 text-right transition hover:border-ink/30 md:col-start-2"
          >
            <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
              Next · step {i + 2} of {NAV.length} &rarr;
            </span>
            <p className="mt-3 font-display text-2xl font-medium text-ink group-hover:text-accent">{next.label}</p>
            <p className="mt-1 text-sm text-muted">{next.blurb}</p>
          </Link>
        )}
      </div>
    </nav>
  );
}
