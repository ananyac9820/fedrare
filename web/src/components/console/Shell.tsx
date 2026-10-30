"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef } from "react";

/* The console is deliberately a different world from the research site: that one is paper
   and daylight and finished findings, this one is a dark instrument panel for a system
   that is running right now. Colours are declared here rather than in the global theme so
   nothing in the published site can be disturbed by them. */

export const C = {
  bg: "#06060a",
  panel: "rgba(255,255,255,0.025)",
  line: "rgba(255,255,255,0.09)",
  lineBright: "rgba(255,255,255,0.18)",
  text: "#f4f3f7",
  muted: "#9a98a8",
  faint: "#5f5d70",
  violet: "#8b7cf6",
  cyan: "#4dd8e8",
  amber: "#f0b254",
  rose: "#f2667e",
  green: "#5fd3a0",
};

/** Drifting particle field. Cheap, pauses off-screen, still under reduced motion. */
export function ParticleField({ density = 60 }: { density?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion() ?? false;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let running = true;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const dots: { x: number; y: number; vx: number; vy: number; r: number; a: number }[] = [];

    const size = () => {
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { width, height };
    };
    let { width, height } = size();

    for (let i = 0; i < density; i++) {
      dots.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
        r: Math.random() * 1.6 + 0.4,
        a: Math.random() * 0.4 + 0.12,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      for (const d of dots) {
        if (!reduced) {
          d.x += d.vx;
          d.y += d.vy;
          if (d.x < 0) d.x = width;
          if (d.x > width) d.x = 0;
          if (d.y < 0) d.y = height;
          if (d.y > height) d.y = 0;
        }
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(190,185,255,${d.a})`;
        ctx.fill();
      }
      // thread nearby dots, which reads as a network rather than confetti
      for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
          const dx = dots[i].x - dots[j].x;
          const dy = dots[i].y - dots[j].y;
          const dist = Math.hypot(dx, dy);
          if (dist < 110) {
            ctx.beginPath();
            ctx.moveTo(dots[i].x, dots[i].y);
            ctx.lineTo(dots[j].x, dots[j].y);
            ctx.strokeStyle = `rgba(150,145,220,${0.07 * (1 - dist / 110)})`;
            ctx.lineWidth = 0.6;
            ctx.stroke();
          }
        }
      }
      if (running && !reduced) raf = requestAnimationFrame(draw);
    };

    const onResize = () => {
      ({ width, height } = size());
    };
    window.addEventListener("resize", onResize);
    draw();
    return () => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, [density, reduced]);

  return <canvas ref={ref} aria-hidden className="absolute inset-0 h-full w-full" />;
}

/** The glow washes that give the dark surface depth. */
export function Ambience() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <div
        className="absolute -left-40 -top-40 h-[32rem] w-[32rem] rounded-full blur-[120px]"
        style={{ background: "radial-gradient(circle, rgba(139,124,246,0.22), transparent 70%)" }}
      />
      <div
        className="absolute -right-32 top-1/3 h-[28rem] w-[28rem] rounded-full blur-[130px]"
        style={{ background: "radial-gradient(circle, rgba(77,216,232,0.14), transparent 70%)" }}
      />
      <div
        className="absolute bottom-0 left-1/3 h-[24rem] w-[24rem] rounded-full blur-[140px]"
        style={{ background: "radial-gradient(circle, rgba(242,102,126,0.10), transparent 70%)" }}
      />
    </div>
  );
}

export function Panel({
  children,
  className = "",
  glow = false,
}: {
  children: React.ReactNode;
  className?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={`relative rounded-2xl border backdrop-blur-xl ${className}`}
      style={{
        borderColor: C.line,
        background: glow
          ? "linear-gradient(145deg, rgba(139,124,246,0.10), rgba(255,255,255,0.02))"
          : C.panel,
      }}
    >
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  accent = C.text,
  delay = 0,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      <Panel className="h-full p-5">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
          {label}
        </p>
        <p className="mt-3 text-3xl font-semibold tabular-nums" style={{ color: accent }}>
          {value}
        </p>
        {sub && (
          <p className="mt-1.5 text-xs leading-relaxed" style={{ color: C.muted }}>
            {sub}
          </p>
        )}
      </Panel>
    </motion.div>
  );
}

/** Horizontal bar used for weights, trust and probabilities. */
export function Meter({
  value,
  reference,
  colour = C.violet,
  height = 8,
}: {
  value: number;
  reference?: number;
  colour?: string;
  height?: number;
}) {
  const reduced = useReducedMotion() ?? false;
  return (
    <div
      className="relative w-full overflow-hidden rounded-full"
      style={{ height, background: "rgba(255,255,255,0.07)" }}
    >
      <motion.div
        className="h-full rounded-full"
        style={{ background: colour, boxShadow: `0 0 14px ${colour}66` }}
        initial={{ width: reduced ? `${Math.min(100, value * 100)}%` : 0 }}
        animate={{ width: `${Math.min(100, value * 100)}%` }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
      />
      {reference !== undefined && (
        <div
          className="absolute top-0 h-full w-px"
          style={{ left: `${Math.min(100, reference * 100)}%`, background: "rgba(255,255,255,0.55)" }}
          title={`reference ${(reference * 100).toFixed(1)}%`}
        />
      )}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "solid",
  disabled,
  type = "button",
  full,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "solid" | "ghost" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
  full?: boolean;
}) {
  const styles: Record<string, React.CSSProperties> = {
    solid: { background: C.violet, color: "#0b0a14", boxShadow: `0 8px 30px -12px ${C.violet}` },
    ghost: { background: "rgba(255,255,255,0.05)", color: C.text, border: `1px solid ${C.line}` },
    danger: { background: "rgba(242,102,126,0.14)", color: C.rose, border: `1px solid ${C.rose}55` },
  };
  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled}
      whileHover={disabled ? undefined : { y: -1 }}
      whileTap={disabled ? undefined : { y: 1 }}
      className={`inline-flex items-center justify-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold transition disabled:opacity-40 ${full ? "w-full" : ""}`}
      style={styles[variant]}
    >
      {children}
    </motion.button>
  );
}

export function Spinner({ label }: { label?: string }) {
  const reduced = useReducedMotion() ?? false;
  return (
    <div className="flex items-center gap-3">
      <motion.span
        aria-hidden
        className="block h-4 w-4 rounded-full border-2 border-t-transparent"
        style={{ borderColor: `${C.violet} transparent ${C.violet} ${C.violet}` }}
        animate={reduced ? {} : { rotate: 360 }}
        transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
      />
      {label && (
        <span className="font-mono text-xs uppercase tracking-[0.18em]" style={{ color: C.muted }}>
          {label}
        </span>
      )}
    </div>
  );
}

export function Empty({ title, body }: { title: string; body: string }) {
  return (
    <Panel className="p-8 text-center">
      <p className="text-base font-medium" style={{ color: C.text }}>
        {title}
      </p>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed" style={{ color: C.muted }}>
        {body}
      </p>
    </Panel>
  );
}
