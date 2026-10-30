"use client";

import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { api, ApiError, pct, type Prediction, type Status, type User } from "@/lib/console-api";
import { Button, C, Empty, Meter, Panel, Spinner, Stat } from "./Shell";
import type { TabDef } from "./Console";

interface Props {
  user: User;
  status: Status | null;
  refresh: () => void;
}

function Diagnose({ status }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [model, setModel] = useState<"sandbox" | "research">("sandbox");
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const rare = status?.rare_classes ?? [];

  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);

  function choose(f: File | undefined) {
    if (!f || !/\.(jpe?g|png)$/i.test(f.name)) {
      setError("Pick a JPEG or PNG image.");
      return;
    }
    if (url) URL.revokeObjectURL(url);
    setFile(f);
    setUrl(URL.createObjectURL(f));
    setOut(null);
    setError(null);
  }

  async function run() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setOut(await api.diagnose(file, model));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The diagnosis failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <div className="grid gap-5">
        <motion.div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            choose(e.dataTransfer.files[0]);
          }}
          animate={{ scale: dragging ? 1.01 : 1 }}
          className="overflow-hidden rounded-2xl border-2 border-dashed"
          style={{
            borderColor: dragging ? C.cyan : C.line,
            background: dragging ? "rgba(77,216,232,0.07)" : "rgba(255,255,255,0.02)",
          }}
        >
          {url ? (
            // eslint-disable-next-line @next/next/no-img-element -- local object URL
            <img src={url} alt="the image being examined" className="aspect-square w-full object-cover" />
          ) : (
            <div className="flex aspect-square flex-col items-center justify-center p-8 text-center">
              <p className="text-base font-medium" style={{ color: C.text }}>
                Drop an image
              </p>
              <p className="mt-2 max-w-xs text-sm leading-relaxed" style={{ color: C.muted }}>
                A dermoscopy photograph. It is examined on this machine and never sent anywhere
                else.
              </p>
              <div className="mt-6">
                <Button variant="ghost" onClick={() => inputRef.current?.click()}>
                  Choose an image
                </Button>
              </div>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg"
            className="sr-only"
            onChange={(e) => {
              choose(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </motion.div>

        <Panel className="p-5">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
            Which model
          </p>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {([
              ["sandbox", "Live system", "Trained by the hospitals signed in here"],
              ["research", "Published model", "The study's own checkpoint, frozen"],
            ] as const).map(([id, title, sub]) => (
              <button
                key={id}
                type="button"
                onClick={() => setModel(id)}
                className="rounded-xl border p-3 text-left transition"
                style={{
                  borderColor: model === id ? C.cyan : C.line,
                  background: model === id ? "rgba(77,216,232,0.08)" : "transparent",
                }}
              >
                <p className="text-sm font-medium" style={{ color: model === id ? C.cyan : C.text }}>
                  {title}
                </p>
                <p className="mt-1 text-xs leading-snug" style={{ color: C.muted }}>
                  {sub}
                </p>
              </button>
            ))}
          </div>
          <div className="mt-5">
            {busy ? <Spinner label="examining" /> : <Button onClick={run} full>{file ? "Diagnose" : "Choose an image first"}</Button>}
          </div>
          {error && (
            <p className="mt-3 text-sm" style={{ color: C.rose }}>
              {error}
            </p>
          )}
        </Panel>
      </div>

      <div className="grid content-start gap-5">
        {out ? (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-5">
            <Panel glow className="p-7">
              <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
                Most likely
              </p>
              <p className="mt-3 text-balance text-3xl font-semibold leading-tight sm:text-4xl" style={{ color: C.text }}>
                {out.predicted.name}
              </p>
              <p className="mt-3 text-sm" style={{ color: C.muted }}>
                {pct(out.predicted.confidence)} confident · next most likely {out.runner_up.name} at{" "}
                {pct(out.runner_up.confidence)} · {out.inference_ms} ms
              </p>
              {rare.includes(out.predicted.name) && (
                <p className="mt-4 rounded-xl px-4 py-3 text-sm" style={{ background: "rgba(139,124,246,0.12)", color: C.violet }}>
                  This is one of the rare classes the system is built around.
                </p>
              )}
            </Panel>

            <Panel className="p-6">
              <p className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: C.faint }}>
                Across all eight diagnoses
              </p>
              <div className="mt-5 grid gap-3">
                {out.probabilities.map((p) => {
                  const top = p.index === out.predicted.index;
                  return (
                    <div key={p.index}>
                      <div className="flex items-baseline justify-between gap-3">
                        <span className="truncate text-sm" style={{ color: top ? C.text : C.muted }}>
                          {p.name}
                        </span>
                        <span className="font-mono text-xs tabular-nums" style={{ color: C.faint }}>
                          {pct(p.probability)}
                        </span>
                      </div>
                      <div className="mt-1.5">
                        <Meter value={p.probability} colour={top ? C.cyan : "rgba(255,255,255,0.22)"} height={6} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-6 text-xs leading-relaxed" style={{ color: C.faint }}>
                A research prototype on public data, not a medical device. It is right about two
                times in five across the eight classes, so treat a confident answer as a suggestion
                and nothing more.
              </p>
            </Panel>
          </motion.div>
        ) : (
          <Empty
            title="No image examined yet"
            body="Drop a dermoscopy image on the left and choose which model should look at it. The live system's model changes as hospitals contribute; the published one never does."
          />
        )}
      </div>
    </div>
  );
}

function History() {
  const [rows, setRows] = useState<Awaited<ReturnType<typeof api.diagnoses>> | null>(null);
  useEffect(() => {
    api.diagnoses().then(setRows).catch(() => setRows([]));
  }, []);

  if (!rows) return <Spinner label="loading" />;
  if (!rows.length) return <Empty title="No history yet" body="Images you examine are listed here, newest first." />;

  return (
    <Panel className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] text-left text-sm">
          <thead>
            <tr style={{ color: C.faint }} className="font-mono text-[10px] uppercase tracking-[0.16em]">
              <th className="px-5 py-3 font-medium">When</th>
              <th className="px-5 py-3 font-medium">Image</th>
              <th className="px-5 py-3 font-medium">Model</th>
              <th className="px-5 py-3 font-medium">Result</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t" style={{ borderColor: C.line }}>
                <td className="px-5 py-3 font-mono text-xs" style={{ color: C.faint }}>
                  {r.created_at.slice(11, 16)} · {r.created_at.slice(0, 10)}
                </td>
                <td className="max-w-[12rem] truncate px-5 py-3" style={{ color: C.muted }}>
                  {r.filename}
                </td>
                <td className="px-5 py-3" style={{ color: r.model === "sandbox" ? C.cyan : C.muted }}>
                  {r.model}
                </td>
                <td className="px-5 py-3" style={{ color: C.text }}>
                  {r.predicted}{" "}
                  <span className="font-mono text-xs" style={{ color: C.faint }}>
                    {pct(r.confidence)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function About({ status }: Props) {
  const s = status;
  return (
    <div className="grid gap-5">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Live model round"
          value={`${s?.sandbox.round ?? 0}`}
          sub="rounds the hospitals have run here"
          accent={C.cyan}
        />
        <Stat
          label="Live rare recall"
          value={s?.sandbox.metrics ? pct(s.sandbox.metrics.rare_recall) : "—"}
          delay={0.05}
        />
        <Stat
          label="Published balanced accuracy"
          value={s ? pct(s.research.metrics.balanced_accuracy) : "—"}
          sub="frozen; the study's number"
          delay={0.1}
        />
        <Stat
          label="Published rare F1"
          value={s ? s.research.metrics.rare_macro_f1.toFixed(3) : "—"}
          delay={0.15}
        />
      </div>
      <Panel className="p-6">
        <p className="max-w-3xl text-sm leading-relaxed" style={{ color: C.muted }}>
          Two models are available to you. The <strong style={{ color: C.cyan }}>live system</strong>{" "}
          model began as a copy of the published one and changes every time hospitals here run a
          round — so its numbers move. The{" "}
          <strong style={{ color: C.text }}>published model</strong> is the study&apos;s own
          checkpoint and never changes, which is what lets the research results stay comparable.
        </p>
      </Panel>
    </div>
  );
}

export function ClinicTabs(props: Props): TabDef[] {
  return [
    { id: "diagnose", label: "Diagnose", render: () => <Diagnose {...props} /> },
    { id: "history", label: "History", render: () => <History /> },
    { id: "about", label: "The models", render: () => <About {...props} /> },
  ];
}
