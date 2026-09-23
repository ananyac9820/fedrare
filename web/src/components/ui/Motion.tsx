"use client";

import { animate, motion, useInView, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { useEffect, useRef } from "react";
import { Pill } from "@/components/ui/Status";

/** true when the visitor asked for reduced motion (false while unknown on first render). */
export function useReducedMotionSafe(): boolean {
  return useReducedMotion() ?? false;
}

/** Quiet fade-and-rise when scrolled into view. Under reduced motion MotionConfig strips the
 *  transform, leaving a plain fade. */
export function Reveal({
  children,
  delay = 0,
  as = "div",
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  as?: "div" | "li" | "section";
  className?: string;
}) {
  const Comp = as === "li" ? motion.li : as === "section" ? motion.section : motion.div;
  return (
    <Comp
      className={className}
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.7, ease: [0.25, 0.8, 0.3, 1], delay }}
    >
      {children}
    </Comp>
  );
}

/** Page-top header: big serif headline over a gently parallaxed graph-paper layer. */
export function PageHeader({
  eyebrow,
  title,
  children,
  tags,
}: {
  eyebrow: string;
  title: React.ReactNode;
  children?: React.ReactNode;
  tags?: React.ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotionSafe();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const gridY = useTransform(scrollYProgress, [0, 1], [0, 60]);

  return (
    <section ref={ref} className="relative overflow-hidden border-b border-line">
      <motion.div aria-hidden className="paper-grid absolute inset-0" style={{ y: reduced ? 0 : gridY }} />
      <div aria-hidden className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-cream to-transparent" />
      <div className="relative mx-auto max-w-6xl px-6 pb-24 pt-40 md:pb-32 md:pt-52">
        <Reveal>
          <div className="flex flex-wrap items-center gap-3">
            <Pill tone="accent">{eyebrow}</Pill>
            {tags}
          </div>
          <h1 className="mt-8 max-w-4xl text-balance font-display text-5xl font-medium leading-[1.02] tracking-tight text-ink md:text-7xl">
            {title}
          </h1>
          {children && (
            <div className="mt-8 max-w-2xl text-pretty text-lg leading-relaxed text-muted md:text-xl">
              {children}
            </div>
          )}
        </Reveal>
      </div>
    </section>
  );
}

/** A generously spaced content section with a heading row. */
export function Section({
  id,
  label,
  title,
  intro,
  tags,
  children,
  tone = "cream",
}: {
  id?: string;
  label?: string;
  title?: React.ReactNode;
  intro?: React.ReactNode;
  tags?: React.ReactNode;
  children: React.ReactNode;
  tone?: "cream" | "sand";
}) {
  return (
    <section id={id} className={`scroll-mt-28 py-24 md:py-36 ${tone === "sand" ? "bg-sand/60" : ""}`}>
      <div className="mx-auto max-w-6xl px-6">
        {(title || label) && (
          <Reveal className="mb-14 max-w-3xl md:mb-20">
            <div className="flex flex-wrap items-center gap-3">
              {label && <Pill>{label}</Pill>}
              {tags}
            </div>
            {title && (
              <h2 className="mt-6 text-balance font-display text-4xl font-medium leading-[1.08] tracking-tight text-ink md:text-6xl">
                {title}
              </h2>
            )}
            {intro && <div className="mt-6 text-pretty text-lg leading-relaxed text-muted">{intro}</div>}
          </Reveal>
        )}
        {children}
      </div>
    </section>
  );
}

/** Animated number that counts up once when scrolled into view. Static under reduced motion.
 *  Writes the text directly (an external-system update), so no React state or re-renders. */
export function CountUp({ to, decimals = 0, suffix = "", className = "" }: {
  to: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const text = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduced = useReducedMotionSafe();
  const armed = useRef(false);
  const final = `${to.toFixed(decimals)}${suffix}`;

  useEffect(() => {
    const el = text.current;
    if (!el) return;
    const show = (v: number) => {
      el.textContent = `${v.toFixed(decimals)}${suffix}`;
    };
    if (reduced) {
      show(to);
      return;
    }
    // Server render and no-JS show the real value. Drop to 0 only while still off screen,
    // then count up when it scrolls in - no visible flash.
    if (!inView) {
      if (!armed.current) {
        show(0);
        armed.current = true;
      }
      return;
    }
    if (!armed.current) return;
    const controls = animate(0, to, { duration: 1.4, ease: [0.16, 1, 0.3, 1], onUpdate: show });
    return () => controls.stop();
  }, [inView, reduced, to, decimals, suffix]);

  return (
    <span ref={ref} className={className} aria-label={final}>
      <span ref={text} aria-hidden>
        {final}
      </span>
    </span>
  );
}

/** Whether an element is on screen - used to pause 3D render loops when scrolled away. */
export function useOnScreen<T extends Element>(margin = "200px") {
  const ref = useRef<T>(null);
  const onScreen = useInView(ref, { margin: margin as `${number}px` });
  return [ref, onScreen] as const;
}
