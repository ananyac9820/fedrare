/**
 * Client for the live system API (server/app.py).
 *
 * The research site reads static JSON; this reads a running server. Keeping them in separate
 * modules keeps the published numbers and the sandbox numbers from ever being confused for
 * one another in the UI.
 */

export const API = process.env.NEXT_PUBLIC_CONSOLE_API ?? "http://127.0.0.1:8100";
const TOKEN_KEY = "fedrare.console.token";

export type Role = "admin" | "hospital" | "clinic";

export interface User {
  username: string;
  role: Role;
  display_name: string;
  hospital_id: number | null;
}

export interface Metrics {
  available?: boolean;
  balanced_accuracy: number;
  accuracy: number;
  rare_macro_f1: number;
  rare_recall: number;
  per_class_f1?: Record<string, number>;
}

export interface Contribution {
  hospital: string;
  hospital_id: number;
  images: number;
  size_share: number;
  rare_row_weight: number;
  trust: Record<string, number>;
  declared: Record<string, number>;
}

export interface Round {
  round: number;
  rule: string;
  created_at: string;
  seconds: number | null;
  n_hospitals: number;
  metrics: Metrics;
  contributions: Contribution[];
}

export interface Status {
  sandbox: { round: number; metrics: Metrics | null; seeded_from?: string };
  research: { metrics: Metrics; training: string; read_only: boolean };
  hospitals: { id: number; name: string; n_submissions: number; n_images: number }[];
  classes: string[];
  rare_classes: string[];
}

export interface Prediction {
  model: string;
  filename?: string;
  predicted: { index: number; name: string; confidence: number };
  runner_up: { index: number; name: string; confidence: number };
  probabilities: { index: number; name: string; probability: number }[];
  inference_ms: number;
}

export const getToken = () => {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
};

export const setToken = (t: string | null) => {
  try {
    if (t) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private browsing - the session just won't persist across reloads */
  }
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}, ms = 120000): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  try {
    const res = await fetch(`${API}${path}`, { ...init, headers, signal: ctl.signal });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(body.error ?? `the server answered ${res.status}`, res.status);
    return body as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  meta: () => request<{ classes: string[]; rare_classes: string[]; demo_accounts: { username: string; role: Role }[] }>("/api/meta"),

  login: async (username: string, password: string) => {
    const out = await request<{ token: string; user: User }>("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    setToken(out.token);
    return out.user;
  },

  logout: async () => {
    try {
      await request("/api/logout", { method: "POST" });
    } finally {
      setToken(null);
    }
  },

  me: () => request<User>("/api/me"),
  status: () => request<Status>("/api/status"),
  rounds: () => request<Round[]>("/api/rounds"),
  submissions: () =>
    request<{ id: number; hospital: string; created_at: string; n_images: number; note: string | null; counts: Record<string, number> }[]>(
      "/api/submissions",
    ),

  submit: (files: File[], labels: Record<string, number>, note: string) => {
    const body = new FormData();
    files.forEach((f) => body.append("images", f));
    body.append("labels", JSON.stringify(labels));
    if (note) body.append("note", note);
    return request<{ submission_id: number; accepted: number; rejected: { file: string; why: string }[]; declared: Record<string, number> }>(
      "/api/submissions",
      { method: "POST", body },
      300000,
    );
  },

  train: (rule: "earn" | "fedavg") =>
    request<Round & { contributions: Contribution[] }>(
      "/api/train",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rule }) },
      600000,
    ),

  diagnose: (file: File, model: "sandbox" | "research") => {
    const body = new FormData();
    body.append("image", file);
    body.append("model", model);
    return request<Prediction & { id: number }>("/api/diagnose", { method: "POST", body }, 180000);
  },

  diagnoses: () =>
    request<{ id: number; created_at: string; model: string; filename: string; predicted: string; confidence: number; who?: string }[]>(
      "/api/diagnoses",
    ),

  adminOverview: () =>
    request<{
      hospitals: { id: number; name: string; n_submissions: number; n_images: number }[];
      users: { id: number; username: string; role: Role; display_name: string; hospital_id: number | null }[];
      rounds: { round: number; rule: string; created_at: string; metrics: Metrics }[];
      totals: { submissions: number; diagnoses: number; hospitals: number };
    }>("/api/admin/overview"),

  adminTrust: () =>
    request<{ round: number; classes: string[]; rare_classes: string[]; rows: { hospital: string; hospital_id: number; trust: Record<string, number>; declared: Record<string, number>; rare_row_weight: number; size_share: number }[] }>(
      "/api/admin/trust",
    ),

  adminReset: () => request<{ ok: boolean; note: string }>("/api/admin/reset", { method: "POST" }),
};

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
