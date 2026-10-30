"use client";

import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  pct,
  type Contribution,
  type Round,
  type Status,
  type User,
} from "@/lib/console-api";
import { Button, C, Empty, Meter, Panel, Spinner, Stat } from "./Shell";
import type { TabDef } from "./Console";

interface Props {
  user: User;
  status: Status | null;
  refresh: () => void;
}

/* ------------------------------------------------------------------ submit */

interface Staged {
  file: File;
  label: number | null;
  url: string;
}

function Submit({ status, refresh }: Props) {
  const [staged, setStaged] = useState<Staged[]>([]);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ accepted: number; declared: Record<string, number> } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const classes = status?.classes ?? [];

  const add = useCallback((files: FileList | File[]) => {
    const next = Array.from(files)
      .filter((f) => /\.(jpe?g|png)$/i.test(f.name))
      .map((f) => ({ file: f, label: null, url: URL.createObjectURL(f) }));
    setStaged((s) => [...s, ...next]);
    setResult(null);
    setError(null);
  }, []);

  useEffect(() => () => staged.forEach((s) => URL.revokeObjectURL(s.url)), [staged]);

  const labelled = staged.filter((s) => s.label !== null);

  async function send() {
    if (!labelled.length) {
      setError("Give each image a diagnosis before submitting - that is the declaration the system audits.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const labels: Record<string, number> = {};
      labelled.forEach((s) => {
        labels[s.file.name] = s.label as number;
      });
      const out = await api.submit(labelled.map((s) => s.file), labels, note);
      setResult({ accepted: out.accepted, declared: out.declared });
      setStaged([]);
      setNote("");
      refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The upload did not go through.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_minmax(0,1fr)]">
      <div className="grid gap-6">
        <motion.div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            add(e.dataTransfer.files);
          }}
          animate={{ scale: dragging ? 1.01 : 1 }}
          className="rounded-2xl border-2 border-dashed p-10 text-center transition"
          style={{
            borderColor: dragging ? C.violet : C.line,
            background: dragging ? "rgba(139,124,246,0.07)" : "rgba(255,255,255,0.02)",
          }}
        >
          <p className="text-lg font-medium" style={{ color: C.text }}>
            Drop labelled cases here
          </p>
          <p className="mx-auto mt-2 max-w-sm text-sm leading-relaxed" style={{ color: C.muted }}>
            JPEG or PNG dermoscopy images. Nothing leaves this machine: the images are turned into
            1,024 numbers and only those take part in training.
          </p>
          <div className="mt-6">
            <Button variant="ghost" onClick={() => inputRef.current?.click()}>
              Choose files
            </Button>
          </div>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept="image/png,image/jpeg"
            className="sr-only"
            onChange={(e) => {
              if (e.target.files) add(e.target.files);
              e.target.value = "";
            }}
          />
        </motion.div>

        {staged.length > 0 && (
          <Panel className="p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm font-medium" style={{ color: C.text }}>
                {staged.length} staged · {labelled.length} labelled
              </p>
              <button
                type="button"
                onClick={() => setStaged([])}
                className="text-xs underline underline-offset-4"
                style={{ color: C.muted }}
              >
                clear
              </button>
            </div>

            <ul className="mt-5 grid gap-3 sm:grid-cols-2">
              {staged.map((s, i) => (
                <motion.li
                  key={s.url}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="flex items-center gap-3 rounded-xl border p-3"
                  style={{ borderColor: C.line, background: "rgba(0,0,0,0.25)" }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- local object URL */}
                  <img src={s.url} alt="" className="h-14 w-14 shrink-0 rounded-lg object-cover" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs" style={{ color: C.muted }}>
                      {s.file.name}
                    </p>
                    <select
                      value={s.label ?? ""}
                      onChange={(e) =>
                        setStaged((prev) =>
                          prev.map((p, j) =>
                            j === i ? { ...p, label: e.target.value === "" ? null : Number(e.target.value) } : p,
                          ),
                        )
                      }
                      className="mt-1.5 w-full rounded-lg border px-2 py-1.5 text-xs outline-none"
                      style={{
                        borderColor: s.label === null ? C.rose + "66" : C.line,
                        background: "rgba(0,0,0,0.4)",
                        color: C.text,
                      }}
                    >
                      <option value="">— diagnosis —</option>
                      {classes.map((c, ci) => (
                        <option key={c} value={ci}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                </motion.li>
              ))}
            </ul>

            <div className="mt-6 grid gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
              <label className="grid gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
                  Note (optional)
                </span>
                <input
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="e.g. referrals from the rare-lesion clinic"
                  className="rounded-xl border px-3 py-2.5 text-sm outline-none"
                  style={{ borderColor: C.line, background: "rgba(0,0,0,0.3)", color: C.text }}
                />
              </label>
              {busy ? <Spinner label="extracting features" /> : <Button onClick={send}>Submit batch</Button>}
            </div>

            {error && (
              <p className="mt-4 text-sm" style={{ color: C.rose }}>
                {error}
              </p>
            )}
          </Panel>
        )}

        {result && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <Panel glow className="p-6">
              <p className="text-sm font-medium" style={{ color: C.green }}>
                {result.accepted} cases accepted
              </p>
              <p className="mt-2 text-sm leading-relaxed" style={{ color: C.muted }}>
                This is now your declaration. Each training round checks whether your updates stay
                consistent with it.
              </p>
              <ul className="mt-4 grid gap-2 sm:grid-cols-2">
                {Object.entries(result.declared).map(([name, n]) => (
                  <li key={name} className="flex items-baseline justify-between gap-3 text-sm">
                    <span style={{ color: C.text }}>{name}</span>
                    <span className="font-mono text-xs" style={{ color: C.muted }}>
                      {n}
                    </span>
                  </li>
                ))}
              </ul>
            </Panel>
          </motion.div>
        )}
      </div>

      <Panel className="h-fit p-6">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
          How this works
        </p>
        <ol className="mt-5 grid gap-5">
          {[
            ["Label every case", "Your labels are a claim about what you hold. The system records it."],
            ["Images stay here", "Each becomes 1,024 numbers. Only those join the round."],
            ["Train together", "Your update is merged with the other hospitals'."],
            ["Earn influence", "Agreement with peers and with your own record builds trust, and trust buys say on a disease."],
          ].map(([title, body], i) => (
            <li key={title} className="flex gap-4">
              <span
                className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-[10px]"
                style={{ background: "rgba(139,124,246,0.15)", color: C.violet }}
              >
                {i + 1}
              </span>
              <div>
                <p className="text-sm font-medium" style={{ color: C.text }}>
                  {title}
                </p>
                <p className="mt-1 text-xs leading-relaxed" style={{ color: C.muted }}>
                  {body}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ train */

function Train({ user, status, refresh }: Props) {
  const [busy, setBusy] = useState(false);
  const [round, setRound] = useState<Round | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rule, setRule] = useState<"earn" | "fedavg">("earn");
  const mine = status?.hospitals.find((h) => h.id === user.hospital_id);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setRound(await api.train(rule));
      refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The round did not run.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6">
      <Panel glow className="p-7">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div>
            <h2 className="text-2xl font-semibold" style={{ color: C.text }}>
              Run a training round
            </h2>
            <p className="mt-2 max-w-xl text-sm leading-relaxed" style={{ color: C.muted }}>
              Every hospital that has uploaded takes part. Each trains locally, the updates are
              merged by the rule you pick, and the result is scored on a held-out set nobody trained
              on.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {(["earn", "fedavg"] as const).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRule(r)}
                className="rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.12em] transition"
                style={{
                  borderColor: rule === r ? C.violet : C.line,
                  color: rule === r ? C.violet : C.muted,
                  background: rule === r ? "rgba(139,124,246,0.12)" : "transparent",
                }}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-7 flex flex-wrap items-center gap-5">
          {busy ? <Spinner label="training" /> : <Button onClick={run}>Start round</Button>}
          <p className="text-xs" style={{ color: C.faint }}>
            {mine ? `You have contributed ${mine.n_images} cases` : "You have not uploaded yet"}
          </p>
        </div>
        {error && (
          <p className="mt-4 text-sm" style={{ color: C.rose }}>
            {error}
          </p>
        )}
      </Panel>

      {round && <RoundResult round={round} highlight={user.hospital_id} />}
    </div>
  );
}

export function RoundResult({ round, highlight }: { round: Round; highlight?: number | null }) {
  const m = round.metrics;
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Round" value={`${round.round}`} sub={`${round.rule} · ${round.seconds ?? "-"}s`} />
        <Stat
          label="Balanced accuracy"
          value={m?.available === false ? "—" : pct(m.balanced_accuracy)}
          accent={C.cyan}
          delay={0.05}
        />
        <Stat
          label="Rare-class recall"
          value={m?.available === false ? "—" : pct(m.rare_recall)}
          accent={C.violet}
          sub="the measure that matters for rare disease"
          delay={0.1}
        />
        <Stat label="Hospitals" value={`${round.n_hospitals}`} delay={0.15} />
      </div>

      <Panel className="overflow-hidden p-6">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
          Who got a say on the rare diseases
        </p>
        <div className="mt-5 grid gap-5">
          {round.contributions.map((c: Contribution, i) => {
            const mine = highlight != null && c.hospital_id === highlight;
            return (
              <motion.div
                key={c.hospital_id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.06 }}
              >
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-sm font-medium" style={{ color: mine ? C.violet : C.text }}>
                    {c.hospital}
                    {mine && <span className="ml-2 font-mono text-[10px] uppercase">you</span>}
                  </p>
                  <p className="font-mono text-xs" style={{ color: C.muted }}>
                    {c.images} cases · weight {pct(c.rare_row_weight)} · fair share {pct(c.size_share)}
                  </p>
                </div>
                <div className="mt-2">
                  <Meter value={c.rare_row_weight} reference={c.size_share} colour={mine ? C.violet : C.cyan} />
                </div>
              </motion.div>
            );
          })}
        </div>
        <p className="mt-6 text-xs leading-relaxed" style={{ color: C.faint }}>
          The white line is the share plain averaging would give. A bar past it means the system
          judged that hospital to know more about these diseases than its size alone implies.
        </p>
      </Panel>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ standing */

function Standing({ user, status }: Props) {
  const [rounds, setRounds] = useState<Round[] | null>(null);
  useEffect(() => {
    api.rounds().then(setRounds).catch(() => setRounds([]));
  }, []);

  if (!rounds) return <Spinner label="loading" />;
  if (!rounds.length)
    return (
      <Empty
        title="No rounds yet"
        body="Once a training round has run, your trust and your influence on each disease appear here."
      />
    );

  const mine = rounds
    .map((r) => ({ round: r.round, c: r.contributions.find((c) => c.hospital_id === user.hospital_id) }))
    .filter((x) => x.c);
  const latest = mine[0]?.c;
  const rare = status?.rare_classes ?? [];

  if (!latest)
    return (
      <Empty
        title="You have not joined a round"
        body="Upload a batch of labelled cases, then run a training round to take part."
      />
    );

  return (
    <div className="grid gap-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Cases contributed" value={`${latest.images}`} />
        <Stat label="Your share of the data" value={pct(latest.size_share)} delay={0.05} />
        <Stat
          label="Your say on rare rows"
          value={pct(latest.rare_row_weight)}
          accent={latest.rare_row_weight >= latest.size_share ? C.green : C.amber}
          sub={latest.rare_row_weight >= latest.size_share ? "above your fair share" : "below your fair share"}
          delay={0.1}
        />
      </div>

      <Panel className="p-6">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
          Trust, by disease
        </p>
        <div className="mt-5 grid gap-4">
          {Object.entries(latest.trust).map(([name, t]) => (
            <div key={name}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm" style={{ color: rare.includes(name) ? C.violet : C.text }}>
                  {name}
                  {rare.includes(name) && <span className="ml-2 font-mono text-[10px]">rare</span>}
                </span>
                <span className="font-mono text-xs tabular-nums" style={{ color: C.muted }}>
                  {t.toFixed(2)} · declared {latest.declared[name] ?? 0}
                </span>
              </div>
              <div className="mt-1.5">
                <Meter value={t} colour={rare.includes(name) ? C.violet : C.cyan} height={6} />
              </div>
            </div>
          ))}
        </div>
        <p className="mt-6 text-xs leading-relaxed" style={{ color: C.faint }}>
          Trust starts at zero and rises slowly - about ten honest rounds to reach full - and falls
          fast when a hospital contradicts its own record. It only applies to a disease once you hold
          at least 20 cases of it, which is the threshold the study fixed in advance.
        </p>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ history */

function History() {
  const [subs, setSubs] = useState<Awaited<ReturnType<typeof api.submissions>> | null>(null);
  useEffect(() => {
    api.submissions().then(setSubs).catch(() => setSubs([]));
  }, []);

  if (!subs) return <Spinner label="loading" />;
  if (!subs.length)
    return <Empty title="Nothing submitted yet" body="Your uploaded batches will be listed here with the declaration each one made." />;

  return (
    <div className="grid gap-4">
      {subs.map((s, i) => (
        <motion.div key={s.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
          <Panel className="p-5">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <p className="text-sm font-medium" style={{ color: C.text }}>
                Batch #{s.id} · {s.n_images} cases
              </p>
              <p className="font-mono text-[11px]" style={{ color: C.faint }}>
                {s.created_at.replace("T", " ").replace("+00:00", " UTC")}
              </p>
            </div>
            {s.note && (
              <p className="mt-2 text-sm" style={{ color: C.muted }}>
                {s.note}
              </p>
            )}
            <div className="mt-4 flex flex-wrap gap-2">
              {Object.entries(s.counts).map(([name, n]) => (
                <span
                  key={name}
                  className="rounded-full border px-3 py-1 text-xs"
                  style={{ borderColor: C.line, color: C.muted }}
                >
                  {name} · {n}
                </span>
              ))}
            </div>
          </Panel>
        </motion.div>
      ))}
    </div>
  );
}

export function HospitalTabs(props: Props): TabDef[] {
  return [
    { id: "submit", label: "Submit cases", render: () => <Submit {...props} /> },
    { id: "train", label: "Train", render: () => <Train {...props} /> },
    { id: "standing", label: "Trust & influence", render: () => <Standing {...props} /> },
    { id: "history", label: "My batches", render: () => <History /> },
  ];
}
