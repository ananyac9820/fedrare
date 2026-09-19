import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the project root to web/ so Next never infers it from a lockfile elsewhere on disk.
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
