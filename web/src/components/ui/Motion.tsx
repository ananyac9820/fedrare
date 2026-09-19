"use client";

import {
  animate,
  motion,
  useInView,
  useReducedMotion,
  useScroll,
  useTransform,
} from "framer-motion";
import { useEffect, useRef } from "react";

/** true when the visitor asked for reduced motion (false while unknown on first render). */
export function useReducedMotionSafe(): boolean {
  return useReducedMotion() ?? false;
}

/** Fade / slide / 3D tilt-in when scrolled into view. Under reduced motion MotionConfig
 *  strips the transforms, leaving a plain fade. */
export function Reveal({
  children,
  delay = 0,
  tilt = false,
  from = "below",
  as = "div",
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  tilt?: boolean;
  from?: "below" | "left" | "right";
  as?: "div" | "li";
  className?: string;
}) {
  const offset = from === "below" ? { y: 40 } : { x: from === "left" ? -48 : 48 };
  const Comp = as === "li" ? motion.li : motion.div;
  return (
    <Comp
      className={className}
      initial={{ opacity: 0, ...offset, rotateX: tilt ? 16 : 0 }}
      whileInView={{ opacity: 1, x: 0, y: 0, rotateX: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay }}
      style={{ transformPerspective: 1100, transformOrigin: "50% 100%" }}
    >
      {children}
    </Comp>
  );
}

/** A page section whose background layer scrolls slower than its content (parallax). */
export function ParallaxSection({
  id,
  children,
  glow = "teal",
  className = "",
}: {
  id: string;
  children: React.ReactNode;
  glow?: "teal" | "amber" | "blue" | "rose";
  className?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotionSafe();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
  const far = useTransform(scrollYProgress, [0, 1], [-120, 120]);
  const near = useTransform(scrollYProgress, [0, 1], [-50, 50]);
  const glowColor = {
    teal: "rgb(45 212 191 / 0.13)",
    amber: "rgb(251 191 36 / 0.10)",
    blue: "rgb(96 165 250 / 0.12)",
    rose: "rgb(251 113 133 / 0.10)",
  }[glow];

  return (
    <section id={id} ref={ref} className={`relative scroll-mt-16 overflow-hidden ${className}`}>
      <motion.div
        aria-hidden
        className="grid-bg pointer-events-none absolute inset-[-15%_0]"
        style={{ y: reduced ? 0 : far }}
      />
      <motion.div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          y: reduced ? 0 : near,
          background: `radial-gradient(600px 400px at 15% 20%, ${glowColor}, transparent 70%), radial-gradient(500px 360px at 85% 80%, ${glowColor}, transparent 70%)`,
        }}
      />
      <div className="relative z-10">{children}</div>
    </section>
  );
}

export function SectionHeading({
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
  return (
    <Reveal tilt className="mb-12 max-w-3xl">
      <p className="mb-3 font-mono text-xs uppercase tracking-[0.25em] text-accent">{eyebrow}</p>
      <h2 className="text-balance font-display text-4xl font-semibold tracking-tight text-white md:text-5xl">
        {title}
      </h2>
      {tags && <div className="mt-4 flex flex-wrap gap-2">{tags}</div>}
      {children && <div className="mt-5 text-lg leading-relaxed text-slate-400">{children}</div>}
    </Reveal>
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
  const format = (v: number) => `${v.toFixed(decimals)}${suffix}`;
  const final = format(to);

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
    const controls = animate(0, to, { duration: 1.6, ease: [0.16, 1, 0.3, 1], onUpdate: show });
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
