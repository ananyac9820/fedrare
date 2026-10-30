"use client";

import { motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { api, ApiError, pct, type Round, type Status, type User } from "@/lib/console-api";
import { Button, C, Empty, Meter, Panel, Spinner, Stat } from "./Shell";
import type { TabDef } from "./Console";
import { RoundResult } from "./Hospital";
import { ClinicTabs } from "./Clinic";

interface Props {
  user: User;
  status: Status | null;
  refresh: () => void;
}

function Overview({ status, refresh }: Props) {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.adminOverview>> | null>(null);
  const [resetting, setResetting] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const load = useCallback(() => {
    api.adminOverview().then(setData).catch(() => setData(null));
  }, []);
  useEffect(load, [load]);

  async function reset() {
    setResetting(true);
    try {
      const out = await api.adminReset();
      setNote(out.note);
      setConfirming(false);
      load();
      refresh();
    } catch (err) {
      setNote(err instanceof ApiError ? err.message : "The reset failed.");
    } finally {
      setResetting(false);
    }
  }

  if (!data) return <Spinner label="loading" />;
  const withData = data.hospitals.filter((h) => h.n_images > 0);

  return (
    <div className="grid gap-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Hospitals contributing" value={`${withData.length} of ${data.hospitals.length}`} accent={C.amber} />
        <Stat label="Batches submitted" value={`${data.totals.submissions}`} delay={0.05} />
        <Stat label="Rounds run" value={`${data.rounds.length}`} delay={0.1} />
        <Stat label="Diagnoses made" value={`${data.totals.diagnoses}`} delay={0.15} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.3fr_minmax(0,1fr)]">
        <Panel className="p-6">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
            Data by hospital
          </p>
          <div className="mt-5 grid gap-4">
            {data.hospitals.map((h, i) => {
              const total = Math.max(1, ...data.hospitals.map((x) => x.n_images));
              return (
                <motion.div key={h.id} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-sm" style={{ color: h.n_images ? C.text : C.faint }}>
                      {h.name}
                    </span>
                    <span className="font-mono text-xs" style={{ color: C.muted }}>
                      {h.n_images} cases · {h.n_submissions} batches
                    </span>
                  </div>
                  <div className="mt-1.5">
                    <Meter value={h.n_images / total} colour={h.n_images ? C.amber : "rgba(255,255,255,0.1)"} height={6} />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </Panel>

        <div className="grid content-start gap-6">
          <Panel className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
              Both models
            </p>
            <div className="mt-5 grid gap-4">
              <div>
                <p className="text-sm" style={{ color: C.cyan }}>
                  Live system · round {status?.sandbox.round ?? 0}
                </p>
                <p className="mt-1 font-mono text-xs" style={{ color: C.muted }}>
                  {status?.sandbox.metrics
                    ? `balanced ${pct(status.sandbox.metrics.balanced_accuracy)} · rare recall ${pct(status.sandbox.metrics.rare_recall)}`
                    : "no round yet"}
                </p>
              </div>
              <div className="border-t pt-4" style={{ borderColor: C.line }}>
                <p className="text-sm" style={{ color: C.text }}>
                  Published · read-only
                </p>
                <p className="mt-1 font-mono text-xs" style={{ color: C.muted }}>
                  {status ? `balanced ${pct(status.research.metrics.balanced_accuracy)} · rare F1 ${status.research.metrics.rare_macro_f1.toFixed(3)}` : "—"}
                </p>
              </div>
            </div>
          </Panel>

          <Panel className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
              Danger zone
            </p>
            <p className="mt-3 text-sm leading-relaxed" style={{ color: C.muted }}>
              Clearing the sandbox throws away the live model and every round run here. Uploaded
              cases are kept, and the published study is untouched either way.
            </p>
            <div className="mt-5">
              {resetting ? (
                <Spinner label="clearing" />
              ) : confirming ? (
                <div className="flex flex-wrap gap-3">
                  <Button variant="danger" onClick={reset}>
                    Yes, clear the sandbox
                  </Button>
                  <Button variant="ghost" onClick={() => setConfirming(false)}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button variant="danger" onClick={() => setConfirming(true)}>
                  Reset sandbox
                </Button>
              )}
            </div>
            {note && (
              <p className="mt-4 text-xs" style={{ color: C.muted }}>
                {note}
              </p>
            )}
          </Panel>
        </div>
      </div>

      <Panel className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[32rem] text-left text-sm">
            <thead>
              <tr className="font-mono text-[10px] uppercase tracking-[0.16em]" style={{ color: C.faint }}>
                <th className="px-5 py-3 font-medium">User</th>
                <th className="px-5 py-3 font-medium">Role</th>
                <th className="px-5 py-3 font-medium">Belongs to</th>
              </tr>
            </thead>
            <tbody>
              {data.users.map((u) => (
                <tr key={u.id} className="border-t" style={{ borderColor: C.line }}>
                  <td className="px-5 py-3" style={{ color: C.text }}>
                    {u.display_name}{" "}
                    <span className="font-mono text-xs" style={{ color: C.faint }}>
                      {u.username}
                    </span>
                  </td>
                  <td className="px-5 py-3" style={{ color: u.role === "admin" ? C.amber : C.muted }}>
                    {u.role}
                  </td>
                  <td className="px-5 py-3" style={{ color: C.muted }}>
                    {data.hospitals.find((h) => h.id === u.hospital_id)?.name ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function TrustMatrix() {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.adminTrust>> | null>(null);
  useEffect(() => {
    api.adminTrust().then(setData).catch(() => setData(null));
  }, []);

  if (!data) return <Spinner label="loading" />;
  if (!data.rows.length)
    return <Empty title="No trust yet" body="The matrix fills in once a training round has run. Trust starts at zero for everyone." />;

  const rare = new Set(data.rare_classes);
  return (
    <div className="grid gap-6">
      <Panel className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[46rem] text-left text-sm">
            <thead>
              <tr className="font-mono text-[10px] uppercase tracking-[0.14em]" style={{ color: C.faint }}>
                <th className="sticky left-0 px-4 py-3 font-medium" style={{ background: C.bg }}>
                  Hospital
                </th>
                {data.classes.map((c) => (
                  <th key={c} className="px-3 py-3 font-medium" style={{ color: rare.has(c) ? C.violet : C.faint }}>
                    {c.split(" ")[0]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.hospital_id} className="border-t" style={{ borderColor: C.line }}>
                  <td className="sticky left-0 px-4 py-3 text-sm" style={{ background: C.bg, color: C.text }}>
                    {row.hospital}
                  </td>
                  {data.classes.map((c) => {
                    const t = row.trust[c] ?? 0;
                    const declared = row.declared[c] ?? 0;
                    return (
                      <td key={c} className="px-3 py-3">
                        <div
                          className="rounded-lg px-2 py-1.5 text-center font-mono text-xs tabular-nums"
                          style={{
                            background:
                              t > 0 ? `rgba(139,124,246,${0.1 + 0.5 * t})` : "rgba(255,255,255,0.03)",
                            color: t > 0 ? C.text : C.faint,
                          }}
                          title={`${declared} cases declared`}
                        >
                          {t.toFixed(2)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel className="p-6">
        <p className="max-w-3xl text-sm leading-relaxed" style={{ color: C.muted }}>
          Each cell is one hospital&apos;s trust on one disease, at round {data.round}. Hover a cell
          to see how many cases that hospital declared. A hospital earns trust on a disease only
          once it holds at least <strong style={{ color: C.text }}>20 cases</strong> of it — the
          threshold the study fixed in advance — and trust then rises about 0.1 a round and halves
          when a hospital contradicts its own record.
        </p>
        <p className="mt-3 max-w-3xl text-xs leading-relaxed" style={{ color: C.faint }}>
          Zeros everywhere usually mean the hospitals have not yet uploaded 20 cases of any one
          disease. That is the mechanism behaving correctly, not a fault.
        </p>
      </Panel>
    </div>
  );
}

function Rounds({ user }: Props) {
  const [rounds, setRounds] = useState<Round[] | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => {
    api.rounds().then(setRounds).catch(() => setRounds([]));
  }, []);

  if (!rounds) return <Spinner label="loading" />;
  if (!rounds.length) return <Empty title="No rounds yet" body="Every federated round run on this system will be listed here." />;

  return (
    <div className="grid gap-4">
      {rounds.map((r) => (
        <div key={r.round}>
          <button
            type="button"
            onClick={() => setOpen(open === r.round ? null : r.round)}
            className="w-full text-left"
          >
            <Panel className="p-5 transition hover:border-white/20">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <p className="text-sm font-medium" style={{ color: C.text }}>
                  Round {r.round}{" "}
                  <span className="font-mono text-[11px] uppercase" style={{ color: C.amber }}>
                    {r.rule}
                  </span>
                </p>
                <p className="font-mono text-xs" style={{ color: C.muted }}>
                  {r.n_hospitals} hospitals · balanced {r.metrics?.available === false ? "—" : pct(r.metrics.balanced_accuracy)} · rare recall{" "}
                  {r.metrics?.available === false ? "—" : pct(r.metrics.rare_recall)}
                </p>
              </div>
            </Panel>
          </button>
          {open === r.round && (
            <div className="mt-4">
              <RoundResult round={r} highlight={user.hospital_id} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function AdminTabs(props: Props): TabDef[] {
  const clinic = ClinicTabs(props);
  return [
    { id: "overview", label: "Overview", render: () => <Overview {...props} /> },
    { id: "trust", label: "Trust matrix", render: () => <TrustMatrix /> },
    { id: "rounds", label: "Rounds", render: () => <Rounds {...props} /> },
    { id: "diagnose", label: "Diagnose", render: clinic[0].render },
    { id: "activity", label: "All activity", render: clinic[1].render },
  ];
}
