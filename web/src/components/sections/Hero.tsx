"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import dynamic from "next/dynamic";
import { useRef } from "react";
import { StatusTag } from "@/components/ui/Status";
import { useOnScreen, useReducedMotionSafe } from "@/components/ui/Motion";
import type { NetworkNode } from "@/components/three/HospitalNetwork";

const HospitalNetwork = dynamic(() => import("@/components/three/HospitalNetwork"), {
  ssr: false,
  loading: () => <div className="h-full w-full" />,
});

export function Hero({ nodes, stats }: {
  nodes: NetworkNode[];
  stats: { value: string; label: string }[];
}) {
  const section = useRef<HTMLElement>(null);
  const reduced = useReducedMotionSafe();
  const [canvasBox, onScreen] = useOnScreen<HTMLDivElement>("0px");
  const { scrollYProgress } = useScroll({ target: section, offset: ["start start", "end start"] });
  // Parallax: the 3D layer drifts down (moves slower than the page), the text moves up faster.
  const canvasY = useTransform(scrollYProgress, [0, 1], [0, 220]);
  const textY = useTransform(scrollYProgress, [0, 1], [0, -140]);
  const fade = useTransform(scrollYProgress, [0, 0.75], [1, 0]);

  return (
    <section ref={section} id="top" className="relative min-h-[100svh] overflow-hidden">
      <div aria-hidden className="grid-bg pointer-events-none absolute inset-0" />
      {/* Behind the text on small screens (dimmed); its own right-hand column on desktop. */}
      <div className="absolute inset-0 opacity-45 md:left-[38%] md:opacity-100">
        <motion.div
          ref={canvasBox}
          className="h-full w-full"
          style={{ y: reduced ? 0 : canvasY, opacity: reduced ? 1 : fade }}
        >
          <HospitalNetwork nodes={nodes} active={onScreen} reduced={reduced} />
        </motion.div>
      </div>
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-gradient-to-r from-ink-950 via-ink-950/70 to-transparent md:via-ink-950/40"
      />
      <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-ink-950 to-transparent" />

      <motion.div
        className="relative z-10 mx-auto flex min-h-[100svh] max-w-7xl flex-col justify-center px-6 pb-24 pt-28"
        style={{ y: reduced ? 0 : textY }}
      >
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="mb-5 font-mono text-xs uppercase tracking-[0.3em] text-accent"
        >
          Federated learning · Blockchain · Medical AI
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-2xl font-display text-6xl font-semibold leading-[0.95] tracking-tight text-white md:text-8xl"
        >
          EARN
          <span className="mt-3 block text-3xl font-medium leading-tight text-slate-300 md:text-5xl">
            Earned trust for rare diseases
          </span>
        </motion.h1>
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.35 }}
          className="mt-8 max-w-xl text-lg leading-relaxed text-slate-300"
        >
          <p>
            Six real hospitals train one skin-disease classifier together, and no patient image
            ever leaves its hospital.
          </p>
          <p className="mt-3 text-slate-400">
            On top of that we&apos;re building a coverage-aware, ledger-anchored way to decide whose
            updates should count for the rarest diseases.{" "}
            <StatusTag status="in-progress" className="align-middle" />
          </p>
        </motion.div>

        <motion.dl
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.55 }}
          className="mt-12 grid max-w-2xl grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4"
        >
          {stats.map((s) => (
            <div key={s.label} className="flex flex-col-reverse border-l border-line pl-4">
              <dt className="mt-1 text-xs leading-snug text-slate-400">{s.label}</dt>
              <dd className="font-display text-3xl font-semibold text-white">{s.value}</dd>
            </div>
          ))}
        </motion.dl>
        <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
          <StatusTag status="verified" /> Fed-ISIC2019: real hospitals, not simulated splits
        </div>

        <div className="mt-10 flex flex-wrap gap-3">
          <a
            href="#problem"
            className="rounded-full bg-accent px-6 py-3 text-sm font-semibold text-ink-950 transition hover:bg-teal-300"
          >
            See the gap
          </a>
          <a
            href="#status"
            className="rounded-full border border-line px-6 py-3 text-sm font-semibold text-slate-200 transition hover:border-slate-400"
          >
            Honest project status
          </a>
        </div>
        <p className="mt-10 max-w-md font-mono text-[11px] leading-relaxed text-slate-500">
          Node size = images held by each hospital (real counts). Moving particles = model updates.
          Images never travel.
        </p>
      </motion.div>
    </section>
  );
}
