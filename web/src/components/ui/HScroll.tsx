"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useReducedMotionSafe } from "@/components/ui/Motion";

/** A horizontally scrolling row of cards with snap points and previous / next buttons.
 *  Children should be fixed-width cards (e.g. w-[82vw] sm:w-[360px]). */
export function HScroll({ children, label }: { children: React.ReactNode; label: string }) {
  const track = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotionSafe();
  const [atStart, setAtStart] = useState(true);
  const [atEnd, setAtEnd] = useState(false);

  const update = useCallback(() => {
    const el = track.current;
    if (!el) return;
    setAtStart(el.scrollLeft <= 4);
    setAtEnd(el.scrollLeft + el.clientWidth >= el.scrollWidth - 4);
  }, []);

  useEffect(() => {
    const el = track.current;
    if (!el) return;
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, [update]);

  const go = (dir: 1 | -1) => {
    const el = track.current;
    if (!el) return;
    el.scrollBy({ left: dir * el.clientWidth * 0.8, behavior: reduced ? "auto" : "smooth" });
  };

  const button =
    "flex h-11 w-11 items-center justify-center rounded-full border border-line bg-paper text-ink shadow-sm transition hover:border-ink/40 disabled:cursor-default disabled:opacity-35";

  return (
    <div role="region" aria-label={label}>
      <div
        ref={track}
        tabIndex={0}
        className="no-scrollbar -mx-6 flex snap-x snap-mandatory gap-6 overflow-x-auto scroll-px-6 px-6 pb-4 outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
      >
        {children}
      </div>
      <div className="mt-8 flex gap-3">
        <button type="button" className={button} onClick={() => go(-1)} disabled={atStart} aria-label={`Previous: ${label}`}>
          <span aria-hidden>←</span>
        </button>
        <button type="button" className={button} onClick={() => go(1)} disabled={atEnd} aria-label={`Next: ${label}`}>
          <span aria-hidden>→</span>
        </button>
      </div>
    </div>
  );
}
