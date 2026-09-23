"use client";

import { MotionConfig } from "framer-motion";

/** reducedMotion="user": framer-motion drops transform animations (keeps fades) whenever the
 *  visitor's OS asks for reduced motion. The 3D scenes and parallax check it themselves. */
export function Providers({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
