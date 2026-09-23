"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import dynamic from "next/dynamic";
import { useRef } from "react";
import type { NetworkNode } from "@/components/three/HospitalNetwork";
import { PillLink } from "@/components/ui/Buttons";
import { useOnScreen, useReducedMotionSafe } from "@/components/ui/Motion";
import { Pill, StatusTag } from "@/components/ui/Status";

const HospitalNetwork = dynamic(() => import("@/components/three/HospitalNetwork"), {
  ssr: false,
  loading: () => <div className="h-full w-full" />,
});

export function HomeHero({ nodes }: { nodes: NetworkNode[] }) {
  const section = useRef<HTMLElement>(null);
  const reduced = useReducedMotionSafe();
  const [canvasBox, onScreen] = useOnScreen<HTMLDivElement>("0px");
  const { scrollYProgress } = useScroll({ target: section, offset: ["start start", "end start"] });
  // Subtle parallax: the grid and the 3D layer drift at different, slower rates than the text.
  const gridY = useTransform(scrollYProgress, [0, 1], [0, 80]);
  const sceneY = useTransform(scrollYProgress, [0, 1], [0, 50]);

  return (
    <section ref={section} className="relative overflow-hidden border-b border-line">
      <motion.div aria-hidden className="paper-grid absolute inset-0" style={{ y: reduced ? 0 : gridY }} />
      <div className="relative mx-auto grid min-h-[100svh] max-w-6xl items-center gap-12 px-6 pb-24 pt-40 lg:grid-cols-[1.05fr_1fr] lg:gap-8 lg:pt-32">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.25, 0.8, 0.3, 1] }}
        >
          <Pill tone="accent">Federated learning · Blockchain · Medical AI</Pill>
          <h1 className="mt-8 font-display text-6xl font-medium leading-[0.98] tracking-tight text-ink md:text-8xl">
            Earned trust for{" "}
            <span className="whitespace-nowrap rounded-xl bg-accent-soft px-3">rare diseases</span>
          </h1>
          <p className="mt-10 max-w-xl text-pretty text-xl leading-relaxed text-muted">
            Six real hospitals train one skin-disease classifier together. No patient image ever
            leaves its hospital.
          </p>
          <p className="mt-5 max-w-xl text-pretty text-lg leading-relaxed text-muted">
            We tested who should get a say on the rarest diseases, and what happens when a hospital
            lies - and we are open about which parts worked and which did not.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-2">
            <StatusTag status="verified" />
            <span className="text-sm text-faint">real hospitals and data</span>
            <span className="mx-1 text-line">|</span>
            <StatusTag status="verified" />
            <span className="text-sm text-faint">the attack study</span>
            <span className="mx-1 text-line">|</span>
            <StatusTag status="failed" />
            <span className="text-sm text-faint">the evidence signal</span>
          </div>
          <div className="mt-12 flex flex-wrap gap-3">
            <PillLink href="/problem">See the problem</PillLink>
            <PillLink href="/status" variant="outline">
              Honest project status
            </PillLink>
          </div>
        </motion.div>

        <motion.div style={{ y: reduced ? 0 : sceneY }} className="relative">
          <div ref={canvasBox} className="h-[380px] w-full md:h-[520px]">
            <HospitalNetwork nodes={nodes} active={onScreen} reduced={reduced} />
          </div>
          <p className="mx-auto mt-2 max-w-sm text-center font-mono text-[11px] leading-relaxed text-faint">
            Node size = images held by each hospital (real counts). Moving dots = model updates.
            Images never travel.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
