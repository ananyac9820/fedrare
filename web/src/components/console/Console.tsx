"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api, getToken, setToken, type Role, type Status, type User } from "@/lib/console-api";
import { Ambience, Button, C, ParticleField, Spinner } from "./Shell";
import { Login } from "./Login";
import { HospitalTabs } from "./Hospital";
import { ClinicTabs } from "./Clinic";
import { AdminTabs } from "./Admin";

export interface TabDef {
  id: string;
  label: string;
  render: () => React.ReactNode;
}

const ROLE_COLOUR: Record<Role, string> = {
  hospital: C.violet,
  clinic: C.cyan,
  admin: C.amber,
};

export function Console() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);
  const [status, setStatus] = useState<Status | null>(null);
  const [tab, setTab] = useState(0);
  const [offline, setOffline] = useState(false);

  // Resume a session if one is in this browser; otherwise show the sign-in screen.
  useEffect(() => {
    let alive = true;
    // State is only ever set from a settled promise here: the React Compiler rule (rightly)
    // rejects a setState that runs synchronously on an effect's own path.
    if (!getToken()) {
      Promise.resolve().then(() => {
        if (alive) setChecking(false);
      });
      return () => {
        alive = false;
      };
    }
    api
      .me()
      .then((u) => {
        if (alive) setUser(u);
      })
      .catch(() => {
        setToken(null);
      })
      .finally(() => {
        if (alive) setChecking(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  const refresh = useCallback(() => {
    api
      .status()
      .then((s) => {
        setStatus(s);
        setOffline(false);
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    if (user) refresh();
  }, [user, refresh]);

  if (checking) {
    return (
      <div
        className="flex min-h-dvh items-center justify-center"
        style={{ background: C.bg }}
      >
        <Spinner label="restoring session" />
      </div>
    );
  }

  if (!user) {
    return (
      <div style={{ background: C.bg, color: C.text }}>
        <Login onSignedIn={setUser} />
      </div>
    );
  }

  const tabs: TabDef[] =
    user.role === "hospital"
      ? HospitalTabs({ user, status, refresh })
      : user.role === "clinic"
        ? ClinicTabs({ user, status, refresh })
        : AdminTabs({ user, status, refresh });

  const safeTab = Math.min(tab, tabs.length - 1);
  const accent = ROLE_COLOUR[user.role];

  return (
    <div className="relative min-h-dvh overflow-x-hidden" style={{ background: C.bg, color: C.text }}>
      <Ambience />
      <div className="pointer-events-none absolute inset-0 h-[60vh] opacity-50">
        <ParticleField density={45} />
      </div>

      {/* top bar */}
      <header className="relative z-20 border-b backdrop-blur-xl" style={{ borderColor: C.line }}>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-5 gap-y-3 px-5 py-4 sm:px-7">
          <Link href="/" className="text-lg font-semibold tracking-tight" style={{ color: C.text }}>
            earn<span style={{ color: accent }}>.</span>
            <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.22em]" style={{ color: C.faint }}>
              console
            </span>
          </Link>

          <div className="ml-auto flex items-center gap-3">
            {offline && (
              <span className="font-mono text-[10px] uppercase tracking-[0.16em]" style={{ color: C.rose }}>
                server unreachable
              </span>
            )}
            <div className="hidden text-right sm:block">
              <p className="text-sm font-medium" style={{ color: C.text }}>
                {user.display_name}
              </p>
              <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: accent }}>
                {user.role}
              </p>
            </div>
            <Button
              variant="ghost"
              onClick={async () => {
                await api.logout();
                setUser(null);
                setTab(0);
              }}
            >
              Sign out
            </Button>
          </div>
        </div>

        {/* tabs */}
        <nav className="mx-auto max-w-7xl px-5 sm:px-7" aria-label="Console sections">
          <div className="flex gap-1 overflow-x-auto pb-px [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            {tabs.map((t, i) => {
              const active = i === safeTab;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTab(i)}
                  aria-current={active ? "page" : undefined}
                  className="relative shrink-0 px-4 py-3 text-sm font-medium transition"
                  style={{ color: active ? C.text : C.muted }}
                >
                  {t.label}
                  {active && (
                    <motion.span
                      layoutId="console-tab"
                      className="absolute inset-x-2 -bottom-px h-0.5 rounded-full"
                      style={{ background: accent, boxShadow: `0 0 12px ${accent}` }}
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </button>
              );
            })}
          </div>
        </nav>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-5 py-8 sm:px-7 sm:py-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={tabs[safeTab]?.id ?? "empty"}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            {tabs[safeTab]?.render()}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer
        className="relative z-10 border-t px-5 py-6 text-center sm:px-7"
        style={{ borderColor: C.line }}
      >
        <p className="text-xs leading-relaxed" style={{ color: C.faint }}>
          Sandbox system. Anything trained here is kept apart from the published study, which stays
          read-only.{" "}
          <Link href="/" className="underline underline-offset-4" style={{ color: C.muted }}>
            Back to the research site
          </Link>
        </p>
      </footer>
    </div>
  );
}
