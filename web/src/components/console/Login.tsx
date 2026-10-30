"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { api, ApiError, type Role, type User } from "@/lib/console-api";
import { Ambience, Button, C, ParticleField, Spinner } from "./Shell";

const ROLE_BLURB: Record<Role, string> = {
  hospital: "Submit labelled cases, join a training round, and see what your data earned you.",
  clinic: "Send an image, get a diagnosis from the federated model.",
  admin: "Every hospital, every round, the trust matrix, and the controls.",
};

const QUICK: { username: string; role: Role; label: string }[] = [
  { username: "hospital1", role: "hospital", label: "Hospital 1" },
  { username: "hospital2", role: "hospital", label: "Hospital 2" },
  { username: "clinic", role: "clinic", label: "Clinic" },
  { username: "admin", role: "admin", label: "Admin" },
];

export function Login({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!username || !password) {
      setError("Both fields are needed.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.login(username, password));
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not reach the server. Start it with: python server/app.py",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="relative flex min-h-dvh items-center justify-center overflow-hidden px-5 py-16">
      <Ambience />
      <div className="absolute inset-0 opacity-70">
        <ParticleField density={70} />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="relative w-full max-w-5xl"
      >
        <div className="grid gap-10 lg:grid-cols-[1.1fr_minmax(0,0.9fr)] lg:gap-16">
          {/* the pitch */}
          <div className="flex flex-col justify-center">
            <p
              className="font-mono text-[11px] uppercase tracking-[0.3em]"
              style={{ color: C.violet }}
            >
              Federated console
            </p>
            <h1
              className="mt-6 text-balance text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl"
              style={{ color: C.text }}
            >
              Six hospitals.
              <br />
              One model.
              <br />
              <span style={{ color: C.violet }}>No shared images.</span>
            </h1>
            <p
              className="mt-7 max-w-md text-pretty text-base leading-relaxed"
              style={{ color: C.muted }}
            >
              Hospitals contribute what they have seen. The system weighs each one by what it can
              show, audits the claim against its record, and never moves a patient image.
            </p>

            <div className="mt-10 grid gap-3 sm:grid-cols-3">
              {(Object.keys(ROLE_BLURB) as Role[]).map((role, i) => (
                <motion.div
                  key={role}
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.25 + i * 0.09, duration: 0.5 }}
                  className="rounded-xl border p-4"
                  style={{ borderColor: C.line, background: "rgba(255,255,255,0.02)" }}
                >
                  <p
                    className="font-mono text-[10px] uppercase tracking-[0.18em]"
                    style={{ color: C.cyan }}
                  >
                    {role}
                  </p>
                  <p className="mt-2 text-xs leading-relaxed" style={{ color: C.muted }}>
                    {ROLE_BLURB[role]}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>

          {/* the form */}
          <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.15, duration: 0.5 }}
            className="rounded-3xl border p-7 backdrop-blur-2xl sm:p-9"
            style={{
              borderColor: C.lineBright,
              background: "linear-gradient(160deg, rgba(139,124,246,0.10), rgba(255,255,255,0.02))",
            }}
          >
            <h2 className="text-xl font-semibold" style={{ color: C.text }}>
              Sign in
            </h2>
            <p className="mt-1.5 text-sm" style={{ color: C.muted }}>
              Use a demonstration account, or your own if one was created for you.
            </p>

            <form onSubmit={submit} className="mt-7 grid gap-4">
              <label className="grid gap-2">
                <span
                  className="font-mono text-[10px] uppercase tracking-[0.18em]"
                  style={{ color: C.faint }}
                >
                  Username
                </span>
                <input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  className="rounded-xl border px-4 py-3 text-sm outline-none transition focus:border-[#8b7cf6]"
                  style={{ borderColor: C.line, background: "rgba(0,0,0,0.3)", color: C.text }}
                  placeholder="hospital1"
                />
              </label>
              <label className="grid gap-2">
                <span
                  className="font-mono text-[10px] uppercase tracking-[0.18em]"
                  style={{ color: C.faint }}
                >
                  Password
                </span>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  className="rounded-xl border px-4 py-3 text-sm outline-none transition focus:border-[#8b7cf6]"
                  style={{ borderColor: C.line, background: "rgba(0,0,0,0.3)", color: C.text }}
                  placeholder="••••••••"
                />
              </label>

              {error && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-sm"
                  style={{ color: C.rose }}
                >
                  {error}
                </motion.p>
              )}

              <div className="mt-2">
                {busy ? (
                  <Spinner label="signing in" />
                ) : (
                  <Button type="submit" full>
                    Enter the console
                  </Button>
                )}
              </div>
            </form>

            <div className="mt-8 border-t pt-6" style={{ borderColor: C.line }}>
              <p
                className="font-mono text-[10px] uppercase tracking-[0.18em]"
                style={{ color: C.faint }}
              >
                Demonstration accounts · password “fedrare”
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {QUICK.map((q) => (
                  <button
                    key={q.username}
                    type="button"
                    onClick={() => {
                      setUsername(q.username);
                      setPassword("fedrare");
                      setError(null);
                    }}
                    className="rounded-full border px-3 py-1.5 text-xs transition hover:border-[#8b7cf6]"
                    style={{ borderColor: C.line, color: C.muted }}
                  >
                    {q.label}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}
