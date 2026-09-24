"use client";

import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/Buttons";
import { Pill } from "@/components/ui/Status";
import { useReducedMotionSafe } from "@/components/ui/Motion";

/** Where the Python backend is. Override with NEXT_PUBLIC_DEMO_API if you moved the port. */
const API = process.env.NEXT_PUBLIC_DEMO_API ?? "http://127.0.0.1:8000";

type Sample = {
  id: string;
  url: string;
  true_label: number;
  true_label_name: string;
  centre: number;
};

type Prediction = {
  predicted: { index: number; name: string; confidence: number };
  runner_up: { index: number; name: string; confidence: number };
  probabilities: { index: number; name: string; probability: number }[];
  preprocessing: { colour_constancy: boolean; note: string };
  inference_ms: number;
  correct?: boolean;
  source:
    | { kind: "sample"; id: string; true_label: number; true_label_name: string; centre: number }
    | { kind: "upload"; filename: string };
};

type Health = {
  ready: boolean;
  model: { training: string; rounds: number; seed: number; rule: string; backbone: string };
  test_metrics: { balanced_accuracy: number; rare_macro_f1: number; accuracy: number };
  classes: string[];
};

type Backend = "checking" | "ready" | "offline";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** fetch that gives up rather than hanging the page if nothing is listening. */
async function ask(path: string, init?: RequestInit, ms = 30000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const res = await fetch(`${API}${path}`, { ...init, signal: ctl.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error ?? `the backend answered ${res.status}`);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

export function DemoConsole() {
  const reduced = useReducedMotionSafe();
  const [backend, setBackend] = useState<Backend>("checking");
  const [health, setHealth] = useState<Health | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ src: string; caption: string } | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const objectUrl = useRef<string | null>(null);

  // Connecting is driven by a counter rather than a callback: bumping `attempt` re-runs the
  // effect, which is how "Check again" and a failed prediction both ask the backend again.
  // State is only ever set from the settled promise, never on the effect's synchronous path.
  const [attempt, setAttempt] = useState(0);
  const reconnect = () => setAttempt((n) => n + 1);

  useEffect(() => {
    let alive = true;
    Promise.all([ask("/health", undefined, 8000), ask("/samples", undefined, 8000)])
      .then(([h, s]) => {
        if (!alive) return;
        setHealth(h);
        setSamples((s.samples as Sample[]) ?? []);
        setBackend("ready");
      })
      .catch(() => {
        if (alive) setBackend("offline");
      });
    return () => {
      alive = false;
    };
  }, [attempt]);

  // Revoke the last preview URL when it is replaced or the page goes away.
  useEffect(
    () => () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [],
  );

  async function runSample(s: Sample) {
    setSelected(s.id);
    setPreview({ src: `${API}${s.url}`, caption: `Held-out test image · hospital ${s.centre}` });
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(
        await ask("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sample_id: s.id }),
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "the prediction failed");
      reconnect();  // it may have stopped; find out rather than leaving a dead page
    } finally {
      setBusy(false);
    }
  }

  async function runUpload(file: File) {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = URL.createObjectURL(file);
    setSelected(null);
    setPreview({ src: objectUrl.current, caption: `Your image · ${file.name}` });
    setBusy(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    body.append("image", file);
    try {
      setResult(await ask("/predict", { method: "POST", body }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "the prediction failed");
      reconnect();  // it may have stopped; find out rather than leaving a dead page
    } finally {
      setBusy(false);
    }
  }

  if (backend === "offline")
    return (
      <BackendDown
        onRetry={() => {
          setBackend("checking");
          reconnect();
        }}
      />
    );

  return (
    <div className="grid gap-8">
      {backend === "checking" && (
        <Card>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-faint">
            Connecting to the model…
          </p>
        </Card>
      )}

      {backend === "ready" && health && (
        <>
          <ModelBar health={health} />

          <Card>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <Pill tone="accent">Pick an image</Pill>
              <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-faint">
                {samples.length} held-out test images · never seen in training
              </p>
            </div>
            <ul className="mt-8 grid grid-cols-2 [&>*]:min-w-0 gap-4 sm:grid-cols-4">
              {samples.map((s) => {
                const active = selected === s.id;
                return (
                  <li key={s.id}>
                    <button
                      type="button"
                      onClick={() => void runSample(s)}
                      disabled={busy}
                      aria-pressed={active}
                      className={`group w-full overflow-hidden rounded-[20px] border bg-paper text-left transition disabled:opacity-60 ${
                        active ? "border-accent ring-2 ring-accent/30" : "border-line hover:border-ink/30"
                      }`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element -- served by the local demo backend, not Next */}
                      <img
                        src={`${API}${s.url}`}
                        alt={`Dermoscopy image, true diagnosis ${s.true_label_name}`}
                        className="aspect-square w-full object-cover"
                      />
                      <span className="block px-4 py-3">
                        <span className="block truncate text-sm font-semibold text-ink">
                          {s.true_label_name}
                        </span>
                        <span className="mt-0.5 block font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
                          hospital {s.centre}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>

            <div className="mt-8 flex flex-wrap items-center gap-4 border-t border-line pt-6">
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-ink/20 bg-paper px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-ink/50">
                Or upload your own image
                <input
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  disabled={busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void runUpload(file);
                    e.target.value = "";
                  }}
                />
              </label>
              <p className="max-w-md text-xs leading-relaxed text-faint">
                An upload gets the same colour-constancy correction the dataset images already
                carry. It is still a research model on dermoscopy images — not a diagnosis.
              </p>
            </div>
          </Card>

          {/* Deliberately no AnimatePresence here. Its exit animation can strand the loading
              card at opacity 0 and, in "wait" mode, the result then never mounts - the page
              sits on "Running the model" forever. Swapping the child outright always shows
              the answer; the entry animation is all this needs. */}
          <div aria-live="polite" className="grid gap-8">
            {busy ? (
              <Working reduced={reduced} />
            ) : error ? (
              <Card>
                <p className="text-sm text-failed">{error}</p>
                <p className="mt-2 text-sm text-muted">
                  Try the image again. If it keeps happening, restart{" "}
                  <code className="font-mono text-xs">python demo/server.py</code>.
                </p>
              </Card>
            ) : result && preview ? (
              <Result result={result} preview={preview} reduced={reduced} />
            ) : (
              <Card dashed>
                <p className="text-sm text-muted">
                  Choose one of the eight images above. The model runs locally on this laptop —
                  frozen DenseNet-121 features, then the head the six hospitals trained together.
                </p>
              </Card>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function ModelBar({ health }: { health: Health }) {
  const m = health.test_metrics;
  const items = [
    ["Balanced accuracy", pct(m.balanced_accuracy)],
    ["Rare-class macro-F1", m.rare_macro_f1.toFixed(3)],
    ["Plain accuracy", pct(m.accuracy)],
    ["Rounds", `${health.model.rounds} · ${health.model.rule}`],
  ] as const;
  return (
    <Card className="bg-sand/50">
      <div className="grid grid-cols-2 [&>*]:min-w-0 gap-6 md:grid-cols-4">
        {items.map(([label, value]) => (
          <div key={label}>
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-faint">{label}</p>
            <p className="mt-2 font-display text-3xl text-ink">{value}</p>
          </div>
        ))}
      </div>
      <p className="mt-6 text-sm leading-relaxed text-muted">
        {health.model.training}, measured on the held-out test set. It is right about two times
        in five across the eight classes, so expect it to miss some — the true label is shown
        next to every prediction.
      </p>
    </Card>
  );
}

function Working({ reduced }: { reduced: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25 }}
    >
      <Card>
        <div className="flex items-center gap-4">
          <motion.span
            aria-hidden
            className="block h-5 w-5 rounded-full border-2 border-accent border-t-transparent"
            animate={reduced ? {} : { rotate: 360 }}
            transition={{ duration: 0.9, repeat: Infinity, ease: "linear" }}
          />
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-muted">
            Running the model…
          </p>
        </div>
        <div className="mt-8 grid gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-4 rounded-full bg-sand" style={{ width: `${70 - i * 14}%` }} />
          ))}
        </div>
      </Card>
    </motion.div>
  );
}

function Result({
  result,
  preview,
  reduced,
}: {
  result: Prediction;
  preview: { src: string; caption: string };
  reduced: boolean;
}) {
  const isSample = result.source.kind === "sample";
  const correct = result.correct === true;
  const top = result.predicted.index;
  const trueIndex = isSample ? (result.source as { true_label: number }).true_label : -1;

  return (
    <motion.div
      initial={{ opacity: 0, y: reduced ? 0 : 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="grid gap-8 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]"
    >
      <Card className="h-full">
        {/* eslint-disable-next-line @next/next/no-img-element -- local demo backend / object URL */}
        <img
          src={preview.src}
          alt="The image being classified"
          className="aspect-square w-full rounded-[20px] object-cover"
        />
        <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
          {preview.caption}
        </p>
        {isSample && (
          <p className="mt-4 text-sm text-muted">
            True diagnosis:{" "}
            <span className="font-semibold text-ink">
              {(result.source as { true_label_name: string }).true_label_name}
            </span>
          </p>
        )}
        <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
          {result.inference_ms} ms on this laptop
        </p>
      </Card>

      <Card className="h-full">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-faint">
          The model says
        </p>
        <p className="mt-3 text-balance font-display text-4xl leading-tight text-ink md:text-5xl">
          {result.predicted.name}
        </p>
        <p className="mt-3 text-lg text-muted">
          {pct(result.predicted.confidence)} confident · next most likely{" "}
          {result.runner_up.name} at {pct(result.runner_up.confidence)}
        </p>

        {isSample && (
          <p
            className={`mt-6 inline-flex rounded-full px-4 py-2 text-sm font-semibold ${
              correct ? "bg-verified-soft text-verified" : "bg-failed-soft text-failed"
            }`}
          >
            {correct
              ? "Matches the true diagnosis"
              : `Does not match — the true class got ${pct(
                  result.probabilities[trueIndex]?.probability ?? 0,
                )}`}
          </p>
        )}

        <div className="mt-10 grid gap-3">
          {result.probabilities.map((p) => {
            const isTop = p.index === top;
            const isTrue = p.index === trueIndex;
            return (
              <div key={p.index} className="grid grid-cols-[minmax(0,1fr)] gap-1.5">
                <div className="flex items-baseline justify-between gap-3">
                  <span
                    className={`truncate text-sm ${isTop ? "font-semibold text-ink" : "text-muted"}`}
                  >
                    {p.name}
                    {isTrue && !isTop && (
                      <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
                        true label
                      </span>
                    )}
                  </span>
                  <span className="font-mono text-xs text-faint">{pct(p.probability)}</span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-sand">
                  <motion.div
                    className={`h-full rounded-full ${isTop ? "bg-accent" : "bg-ink/25"}`}
                    initial={{ width: reduced ? `${p.probability * 100}%` : 0 }}
                    animate={{ width: `${p.probability * 100}%` }}
                    transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <p className="mt-8 border-t border-line pt-5 text-xs leading-relaxed text-faint">
          {result.preprocessing.note}. Confidences are the softmax over the eight classes and sum
          to 100%.
        </p>
      </Card>
    </motion.div>
  );
}

function BackendDown({ onRetry }: { onRetry: () => void }) {
  return (
    <Card dashed>
      <Pill>Backend not running</Pill>
      <p className="mt-6 font-display text-3xl text-ink">Start the model server first.</p>
      <p className="mt-4 max-w-2xl text-muted">
        The demo runs the real trained model on this machine, so it needs the Python backend
        listening on {API}. In a terminal, from the repository root:
      </p>
      <pre className="mt-6 overflow-x-auto rounded-[20px] border border-line bg-cream p-6 font-mono text-sm text-ink">
        <code>python demo/server.py</code>
      </pre>
      <p className="mt-4 text-sm text-muted">
        The first run needs <code className="font-mono text-xs">python demo/prepare_demo.py</code>{" "}
        once, which trains the head and writes the sample images.
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-8 inline-flex items-center gap-2 rounded-full bg-ink px-6 py-3 text-sm font-semibold text-cream transition hover:bg-accent"
      >
        Check again
      </button>
    </Card>
  );
}
