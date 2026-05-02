"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CircleGauge,
  Database,
  FileSearch,
  Gavel,
  Loader2,
  LockKeyhole,
  Moon,
  PanelRight,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
  XCircle,
} from "lucide-react";

type Role = "public" | "auditor" | "manager" | "executive";
type ThemeMode = "dark" | "light";
type QueryState = "idle" | "processing" | "answer" | "refusal" | "error";

type SourceTrace = {
  chunk_id?: string;
  document_name?: string;
  document_type?: string;
  source_path?: string;
  source_url?: string;
  page_number?: number;
  clause_type?: string;
  similarity_score?: number;
  authority?: string;
};

type StreamEvent =
  | {
      type: "status";
      status: string;
      request_id?: string;
    }
  | {
      type: "chunk";
      content: string;
      index: number;
      queryId?: string;
      query_id?: string;
    }
  | {
      type: "complete";
      decision?: string;
      sources?: string[];
      queryId?: string;
      query_id?: string;
      risk?: string;
      trust_score?: number;
      faithfulness_score?: number;
      refusal_reason?: string | null;
      retrieval_metadata?: SourceTrace[];
    }
  | {
      type: "refusal";
      reason: string;
      risk?: string;
      trust_score?: number;
      faithfulness_score?: number;
      refusal_reason?: string | null;
      retrieval_metadata?: SourceTrace[];
    }
  | {
      type: "error";
      message: string;
      request_id?: string;
    };

type VectorStats = {
  backend?: string;
  collection_name?: string;
  total_chunks?: number;
  storage_target?: string;
};

type HealthState = {
  backend: "checking" | "online" | "offline";
  chroma: "checking" | "online" | "offline";
  documents: "checking" | "online" | "offline";
  vectorStats: VectorStats | null;
  documentsCount: number | null;
  checkedAt: string | null;
};

type QueryResult = {
  state: QueryState;
  question: string;
  answer: string;
  status: string;
  queryId?: string;
  requestId?: string;
  risk?: string;
  trustScore?: number;
  faithfulnessScore?: number;
  refusalReason?: string | null;
  sources: string[];
  traces: SourceTrace[];
  error?: string;
};

const apiBaseUrl = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

const roles: Record<
  Role,
  {
    label: string;
    summary: string;
    clearance: string;
    canAccess: string[];
    likelyRefuses: string[];
  }
> = {
  public: {
    label: "Public",
    summary: "Lowest clearance. Public-safe answers only.",
    clearance: "Public corpus",
    canAccess: ["Public summaries", "Non-sensitive policy facts"],
    likelyRefuses: ["Confidential NDA clauses", "Restricted financial terms"],
  },
  auditor: {
    label: "Auditor",
    summary: "Evidence-first access for compliance review.",
    clearance: "Audit evidence",
    canAccess: ["Clause references", "Controls and review trails"],
    likelyRefuses: ["Executive-only strategy", "Unverified claims"],
  },
  manager: {
    label: "Manager",
    summary: "Operational access for team-level decisions.",
    clearance: "Departmental knowledge",
    canAccess: ["Operational clauses", "Managed team knowledge"],
    likelyRefuses: ["Board-only details", "Low-confidence answers"],
  },
  executive: {
    label: "Executive",
    summary: "Broadest role for high-level business decisions.",
    clearance: "Executive knowledge",
    canAccess: ["Strategic summaries", "High-authority documents"],
    likelyRefuses: ["Unsupported statements", "Faithfulness failures"],
  },
};

const emptyResult: QueryResult = {
  state: "idle",
  question: "",
  answer: "",
  status: "Ready",
  sources: [],
  traces: [],
};

const cn = (...classes: Array<string | false | null | undefined>) =>
  classes.filter(Boolean).join(" ");

const formatPercent = (value?: number) =>
  typeof value === "number" ? `${Math.round(value * 100)}%` : "Not returned";

const formatRisk = (risk?: string) =>
  risk ? risk.replaceAll("_", " ").replace(/^\w/, (letter) => letter.toUpperCase()) : "Reviewed";

const formatLabel = (value?: string | null) =>
  value ? value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()) : "Not specified";

const normalizeQueryId = (event: { queryId?: string; query_id?: string }) =>
  event.queryId ?? event.query_id;

const appendChunk = (current: string, next: string) => {
  if (!current) return next;
  if (!next || /\s$/.test(current) || /^[\s,.;:!?)]/.test(next)) return `${current}${next}`;
  return `${current} ${next}`;
};

const sourceName = (trace: SourceTrace | undefined, fallback: string, index: number) =>
  trace?.document_name || fallback || `Evidence ${index + 1}`;

function ScoreRing({
  label,
  value,
  tone = "cyan",
}: {
  label: string;
  value?: number;
  tone?: "cyan" | "emerald" | "amber";
}) {
  const percent = typeof value === "number" ? Math.max(0, Math.min(100, Math.round(value * 100))) : 0;
  const color =
    tone === "emerald" ? "#10b981" : tone === "amber" ? "#f59e0b" : "#06b6d4";

  return (
    <div className="flex items-center gap-3 rounded-2xl border border-slate-200/60 bg-white/70 p-3 shadow-sm transition dark:border-white/10 dark:bg-white/[0.04]">
      <div
        className="grid h-12 w-12 place-items-center rounded-full text-xs font-semibold text-slate-900 dark:text-white"
        style={{
          background: `conic-gradient(${color} ${percent * 3.6}deg, rgba(148,163,184,.22) 0deg)`,
        }}
      >
        <div className="grid h-9 w-9 place-items-center rounded-full bg-white dark:bg-[#10131a]">
          {typeof value === "number" ? percent : "--"}
        </div>
      </div>
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">{label}</p>
        <p className="text-sm font-medium text-slate-950 dark:text-white">{formatPercent(value)}</p>
      </div>
    </div>
  );
}

function StatusPill({
  state,
  label,
}: {
  state: "checking" | "online" | "offline";
  label: string;
}) {
  const isOnline = state === "online";
  const isChecking = state === "checking";

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition",
        isOnline &&
          "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200",
        isChecking &&
          "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-200",
        state === "offline" &&
          "border-rose-500/25 bg-rose-500/10 text-rose-700 dark:text-rose-200",
      )}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          isOnline && "bg-emerald-500 shadow-[0_0_14px_rgba(16,185,129,.75)]",
          isChecking && "bg-amber-500",
          state === "offline" && "bg-rose-500",
        )}
      />
      {label}
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Database;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200/70 bg-white/75 p-4 shadow-sm transition dark:border-white/10 dark:bg-white/[0.045]">
      <div className="mb-4 flex items-center justify-between">
        <div className="grid h-10 w-10 place-items-center rounded-xl border border-slate-200 bg-slate-50 text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
          <Icon className="h-4 w-4" />
        </div>
      </div>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">{value}</p>
      <p className="mt-1 truncate text-sm text-slate-500 dark:text-slate-400">{detail}</p>
    </div>
  );
}

export default function Home() {
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [role, setRole] = useState<Role>("auditor");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult>(emptyResult);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [themeReady, setThemeReady] = useState(false);
  const [health, setHealth] = useState<HealthState>({
    backend: "checking",
    chroma: "checking",
    documents: "checking",
    vectorStats: null,
    documentsCount: null,
    checkedAt: null,
  });

  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const refreshInFlightRef = useRef(false);

  const isDark = theme === "dark";
  const selectedRole = roles[role];
  const evidence: SourceTrace[] = result.traces.length
    ? result.traces
    : result.sources.map((source) => ({ document_name: source }));

  const lastQueryStatus = useMemo(() => {
    if (result.state === "answer") return "Answered";
    if (result.state === "refusal") return "Refused";
    if (result.state === "error") return "Failed";
    if (result.state === "processing") return "Processing";
    return "No query yet";
  }, [result.state]);

  const hasIndexedData =
    (health.vectorStats?.total_chunks ?? 0) > 0 || (health.documentsCount ?? 0) > 0;

  const systemState = useMemo(() => {
    if (health.backend === "checking" || health.chroma === "checking" || health.documents === "checking") {
      return {
        tone: "checking",
        headline: "Checking governed RAG services",
        detail: "Backend connectivity, vector database, and data presence are being verified automatically.",
      };
    }

    if (!apiBaseUrl) {
      return {
        tone: "offline",
        headline: "API connection is not configured",
        detail: "NEXT_PUBLIC_API_URL is required before governed queries can run from this dashboard.",
      };
    }

    if (health.backend === "offline") {
      return {
        tone: "offline",
        headline: "Backend is unreachable",
        detail: "Governed queries are paused until the Azure-hosted API responds to health checks.",
      };
    }

    if (health.chroma === "offline") {
      return {
        tone: "degraded",
        headline: "Vector database is not ready",
        detail: "Answers may be unavailable because the retrieval index is not responding.",
      };
    }

    if (health.documents === "offline" || !hasIndexedData) {
      return {
        tone: "degraded",
        headline: "Knowledge data is not visible",
        detail: "The backend is online, but indexed chunks or source documents are not currently reported.",
      };
    }

    return {
      tone: "ready",
      headline: "System ready for governed queries",
      detail: "Backend, vector database, and indexed knowledge are all visible inside this command center.",
    };
  }, [hasIndexedData, health.backend, health.chroma, health.documents]);

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("governed-rag-theme");
    const nextTheme =
      savedTheme === "light" || savedTheme === "dark"
        ? savedTheme
        : window.matchMedia("(prefers-color-scheme: light)").matches
          ? "light"
          : "dark";

    setTheme(nextTheme);
    setThemeReady(true);
  }, []);

  useEffect(() => {
    if (!themeReady) return;
    document.documentElement.classList.toggle("dark", isDark);
    document.documentElement.style.colorScheme = isDark ? "dark" : "light";
    window.localStorage.setItem("governed-rag-theme", theme);
  }, [isDark, theme, themeReady]);

  const toggleTheme = () => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  };

  const refreshStatus = useCallback(async () => {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    setIsRefreshing(true);

    if (!apiBaseUrl) {
      setHealth({
        backend: "offline",
        chroma: "offline",
        documents: "offline",
        vectorStats: null,
        documentsCount: null,
        checkedAt: new Date().toLocaleTimeString(),
      });
      refreshInFlightRef.current = false;
      setIsRefreshing(false);
      return;
    }

    setHealth((current) => ({
      ...current,
      backend: "checking",
      chroma: "checking",
      documents: "checking",
    }));

    try {
      const [backendResult, vectorResult, documentsResult] = await Promise.allSettled([
        fetch(`${apiBaseUrl}/health`, { cache: "no-store" }),
        fetch(`${apiBaseUrl}/vector-stats`, { cache: "no-store" }),
        fetch(`${apiBaseUrl}/documents`, { cache: "no-store" }),
      ]);

      let vectorStats: VectorStats | null = null;
      let documentsCount: number | null = null;

      if (vectorResult.status === "fulfilled" && vectorResult.value.ok) {
        vectorStats = (await vectorResult.value.json()) as VectorStats;
      }

      if (documentsResult.status === "fulfilled" && documentsResult.value.ok) {
        const payload = (await documentsResult.value.json()) as { documents?: string[] };
        documentsCount = payload.documents?.length ?? 0;
      }

      setHealth({
        backend:
          backendResult.status === "fulfilled" && backendResult.value.ok ? "online" : "offline",
        chroma: vectorStats ? "online" : "offline",
        documents: documentsCount !== null ? "online" : "offline",
        vectorStats,
        documentsCount,
        checkedAt: new Date().toLocaleTimeString(),
      });
    } catch {
      setHealth((current) => ({
        ...current,
        backend: "offline",
        chroma: "offline",
        documents: "offline",
        checkedAt: new Date().toLocaleTimeString(),
      }));
    } finally {
      refreshInFlightRef.current = false;
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const intervalId = window.setInterval(() => {
      void refreshStatus();
    }, 7000);

    return () => {
      window.clearInterval(intervalId);
      abortRef.current?.abort();
    };
  }, [refreshStatus]);

  const clearQuery = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setQuestion("");
    setIsQuerying(false);
    setResult(emptyResult);
    inputRef.current?.focus();
  };

  const applyEvent = (event: StreamEvent) => {
    if (event.type === "status") {
      setResult((current) => ({
        ...current,
        state: "processing",
        status: formatLabel(event.status),
        requestId: event.request_id ?? current.requestId,
      }));
      return;
    }

    if (event.type === "chunk") {
      setResult((current) => ({
        ...current,
        state: "processing",
        status: "Streaming answer",
        queryId: normalizeQueryId(event) ?? current.queryId,
        answer: appendChunk(current.answer, event.content),
      }));
      return;
    }

    if (event.type === "complete") {
      setResult((current) => ({
        ...current,
        state: "answer",
        status: "Answer complete",
        queryId: normalizeQueryId(event) ?? current.queryId,
        risk: event.risk,
        trustScore: event.trust_score,
        faithfulnessScore: event.faithfulness_score,
        refusalReason: null,
        sources: Array.from(new Set((event.sources ?? []).filter(Boolean))),
        traces: event.retrieval_metadata ?? [],
        answer: current.answer || "The service returned an answer with no streamed text.",
      }));
      return;
    }

    if (event.type === "refusal") {
      setResult((current) => ({
        ...current,
        state: "refusal",
        status: "Access denied",
        risk: event.risk,
        trustScore: event.trust_score,
        faithfulnessScore: event.faithfulness_score,
        refusalReason: event.refusal_reason,
        traces: event.retrieval_metadata ?? [],
        sources: [],
        answer: event.reason,
      }));
      return;
    }

    setResult((current) => ({
      ...current,
      state: "error",
      status: "Query failed",
      requestId: event.request_id ?? current.requestId,
      error: event.message,
      answer: event.message,
    }));
  };

  const processSseLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) return;

    try {
      applyEvent(JSON.parse(trimmed.replace(/^data:\s*/, "")) as StreamEvent);
    } catch {
      setResult((current) => ({
        ...current,
        state: "error",
        status: "Stream parse failed",
        error: "The response stream contained an unreadable event.",
      }));
    }
  };

  const submitQuery = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isQuerying) return;

    if (!apiBaseUrl) {
      setResult({
        ...emptyResult,
        state: "error",
        question: trimmedQuestion,
        status: "Missing API URL",
        error: "NEXT_PUBLIC_API_URL is not configured.",
        answer: "NEXT_PUBLIC_API_URL is not configured.",
      });
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setIsQuerying(true);
    setResult({
      ...emptyResult,
      state: "processing",
      question: trimmedQuestion,
      status: "Submitting governed query",
    });

    try {
      const response = await fetch(`${apiBaseUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          user_role: role,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        lines.forEach(processSseLine);
      }

      if (buffer.trim()) buffer.split("\n").forEach(processSseLine);
    } catch (error) {
      if (!controller.signal.aborted) {
        const message = error instanceof Error ? error.message : "The query could not be completed.";
        setResult((current) => ({
          ...current,
          state: "error",
          status: "Query failed",
          error: message,
          answer: message,
        }));
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setIsQuerying(false);
      void refreshStatus();
    }
  };

  return (
    <main className={cn(isDark ? "dark" : "", "min-h-screen")}>
      <div className="min-h-screen bg-[#f5f7fb] text-slate-950 transition-colors duration-300 dark:bg-[#080b12] dark:text-white">
        <div className="fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(6,182,212,.18),transparent_34%),linear-gradient(135deg,rgba(15,23,42,.02),rgba(15,23,42,.08))] dark:bg-[radial-gradient(circle_at_top_left,rgba(6,182,212,.18),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(16,185,129,.12),transparent_32%),linear-gradient(135deg,#080b12,#0d111a_55%,#090d14)]" />

        <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/80 backdrop-blur-2xl dark:border-white/10 dark:bg-[#080b12]/80">
          <div className="mx-auto flex max-w-[1800px] flex-col gap-4 px-4 py-4 lg:flex-row lg:items-center lg:justify-between lg:px-6">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl border border-cyan-500/25 bg-cyan-500/10 text-cyan-700 shadow-sm dark:text-cyan-200">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-500 dark:text-slate-400">
                  Governed Cloud Hosted RAG
                </p>
                <h1 className="truncate text-lg font-semibold tracking-tight sm:text-xl">
                  Enterprise Knowledge Command Center
                </h1>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <StatusPill state={health.backend} label="Azure backend" />
              <StatusPill state={health.chroma} label="Chroma data" />
              <select
                value={role}
                onChange={(event) => setRole(event.target.value as Role)}
                className="h-9 rounded-full border border-slate-200 bg-white px-3 text-sm font-medium text-slate-800 outline-none transition hover:border-slate-300 dark:border-white/10 dark:bg-white/5 dark:text-white"
                aria-label="Select user role"
              >
                {Object.entries(roles).map(([key, value]) => (
                  <option key={key} value={key}>
                    {value.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={toggleTheme}
                className="grid h-9 w-9 place-items-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:border-cyan-300 hover:text-cyan-700 focus:outline-none focus:ring-2 focus:ring-cyan-400/40 dark:border-white/10 dark:bg-white/5 dark:text-slate-200 dark:hover:bg-white/10"
                aria-label={`Switch to ${isDark ? "light" : "dark"} theme`}
                title={`Switch to ${isDark ? "light" : "dark"} theme`}
              >
                {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
            </div>
          </div>
        </header>

        <section className="mx-auto max-w-[1800px] px-4 pt-4 lg:px-6">
          <div
            className={cn(
              "flex flex-col gap-4 rounded-2xl border p-4 shadow-sm backdrop-blur-xl md:flex-row md:items-center md:justify-between",
              systemState.tone === "ready" &&
                "border-emerald-500/25 bg-emerald-500/10 text-emerald-950 dark:text-emerald-50",
              systemState.tone === "checking" &&
                "border-cyan-500/25 bg-cyan-500/10 text-cyan-950 dark:text-cyan-50",
              systemState.tone === "degraded" &&
                "border-amber-500/25 bg-amber-500/10 text-amber-950 dark:text-amber-50",
              systemState.tone === "offline" &&
                "border-rose-500/25 bg-rose-500/10 text-rose-950 dark:text-rose-50",
            )}
          >
            <div className="flex min-w-0 items-start gap-3">
              <div className="mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-current/20 bg-white/45 dark:bg-black/10">
                {systemState.tone === "ready" ? (
                  <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-300" />
                ) : systemState.tone === "checking" ? (
                  <Loader2 className="h-5 w-5 animate-spin text-cyan-600 dark:text-cyan-300" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-300" />
                )}
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold">{systemState.headline}</p>
                <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">
                  {systemState.detail}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center sm:min-w-[360px]">
              <div className="rounded-xl border border-current/10 bg-white/45 px-3 py-2 dark:bg-black/10">
                <p className="text-xs text-slate-500 dark:text-slate-400">Backend</p>
                <p className="mt-1 text-sm font-semibold capitalize">{health.backend}</p>
              </div>
              <div className="rounded-xl border border-current/10 bg-white/45 px-3 py-2 dark:bg-black/10">
                <p className="text-xs text-slate-500 dark:text-slate-400">Vector DB</p>
                <p className="mt-1 text-sm font-semibold capitalize">{health.chroma}</p>
              </div>
              <div className="rounded-xl border border-current/10 bg-white/45 px-3 py-2 dark:bg-black/10">
                <p className="text-xs text-slate-500 dark:text-slate-400">Chunks</p>
                <p className="mt-1 text-sm font-semibold">
                  {health.vectorStats?.total_chunks?.toLocaleString() ?? "--"}
                </p>
              </div>
            </div>
          </div>
        </section>

        <div className="mx-auto grid max-w-[1800px] gap-4 px-4 py-4 lg:grid-cols-[300px_minmax(0,1fr)_360px] lg:px-6">
          <aside className="space-y-4">
            <section className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-xl shadow-slate-200/50 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.055] dark:shadow-black/20">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">Governance</p>
                  <h2 className="mt-1 text-xl font-semibold tracking-tight">{selectedRole.label} Role</h2>
                </div>
                <Gavel className="h-5 w-5 text-cyan-600 dark:text-cyan-300" />
              </div>
              <select
                value={role}
                onChange={(event) => setRole(event.target.value as Role)}
                className="h-12 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 text-sm font-medium text-slate-900 outline-none transition focus:border-cyan-400 dark:border-white/10 dark:bg-white/5 dark:text-white"
              >
                {Object.entries(roles).map(([key, value]) => (
                  <option key={key} value={key}>
                    {value.label}
                  </option>
                ))}
              </select>
              <p className="mt-4 text-sm leading-6 text-slate-600 dark:text-slate-300">{selectedRole.summary}</p>
              <div className="mt-5 rounded-2xl border border-cyan-500/20 bg-cyan-500/10 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-cyan-700 dark:text-cyan-200">Clearance</p>
                <p className="mt-2 font-semibold text-slate-950 dark:text-white">{selectedRole.clearance}</p>
              </div>
            </section>

            <section className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-xl shadow-slate-200/50 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.055] dark:shadow-black/20">
              <div className="mb-4 flex items-center gap-2">
                <LockKeyhole className="h-4 w-4 text-emerald-600 dark:text-emerald-300" />
                <h3 className="font-semibold">Access Policy</h3>
              </div>
              <div className="space-y-3">
                {selectedRole.canAccess.map((item) => (
                  <div key={item} className="flex items-start gap-3 text-sm text-slate-600 dark:text-slate-300">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                    <span>{item}</span>
                  </div>
                ))}
                {selectedRole.likelyRefuses.map((item) => (
                  <div key={item} className="flex items-start gap-3 text-sm text-slate-600 dark:text-slate-300">
                    <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-500" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-xl shadow-slate-200/50 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.055] dark:shadow-black/20">
              <div className="mb-4 flex items-center gap-2">
                <CircleGauge className="h-4 w-4 text-cyan-600 dark:text-cyan-300" />
                <h3 className="font-semibold">Trust Summary</h3>
              </div>
              <div className="space-y-3">
                <ScoreRing label="Trust" value={result.trustScore} />
                <ScoreRing label="Faithfulness" value={result.faithfulnessScore} tone="emerald" />
              </div>
              <div className="mt-4 rounded-2xl border border-slate-200/70 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/[0.04]">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Current risk</p>
                <p className="mt-2 text-lg font-semibold">{formatRisk(result.risk)}</p>
              </div>
            </section>
          </aside>

          <section className="min-w-0 space-y-4">
            <section className="overflow-hidden rounded-2xl border border-slate-200/70 bg-white/85 shadow-xl shadow-slate-200/60 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.06] dark:shadow-black/20">
              <div className="border-b border-slate-200/70 p-5 dark:border-white/10">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                  <div>
                    <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-700 dark:text-cyan-200">
                      <Sparkles className="h-3.5 w-3.5" />
                      {result.status}
                    </div>
                    <h2 className="mt-4 text-2xl font-semibold tracking-tight sm:text-3xl">Ask the governed knowledge layer</h2>
                    <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-300">
                      The active role is <span className="font-semibold text-slate-900 dark:text-white">{selectedRole.label}</span>. The answer, refusal, risk, and evidence stream into one review surface.
                    </p>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-white/10 dark:bg-white/5">
                      <p className="text-xs text-slate-500 dark:text-slate-400">Mode</p>
                      <p className="text-sm font-semibold">Live stream</p>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-white/10 dark:bg-white/5">
                      <p className="text-xs text-slate-500 dark:text-slate-400">Role</p>
                      <p className="text-sm font-semibold">{selectedRole.label}</p>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 dark:border-white/10 dark:bg-white/5">
                      <p className="text-xs text-slate-500 dark:text-slate-400">State</p>
                      <p className="text-sm font-semibold">{lastQueryStatus}</p>
                    </div>
                  </div>
                </div>

                <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-3 transition focus-within:border-cyan-400 dark:border-white/10 dark:bg-[#0d1118]">
                  <textarea
                    ref={inputRef}
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void submitQuery();
                      }
                    }}
                    disabled={isQuerying}
                    rows={4}
                    placeholder="Ask a question about the governed enterprise corpus..."
                    className="min-h-28 w-full resize-none bg-transparent px-2 py-2 text-base leading-7 text-slate-950 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed disabled:text-slate-400 dark:text-white dark:placeholder:text-slate-500"
                  />
                  <div className="flex flex-col gap-3 border-t border-slate-200 pt-3 dark:border-white/10 sm:flex-row sm:items-center sm:justify-between">
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Live governed response using the active role and Azure-hosted knowledge backend.
                    </p>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={clearQuery}
                        className="inline-flex h-11 items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 text-sm font-medium text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-400/35 dark:border-white/10 dark:bg-white/5 dark:text-slate-200 dark:hover:bg-white/10"
                      >
                        <X className="h-4 w-4" />
                        {isQuerying ? "Stop" : "Clear"}
                      </button>
                      <button
                        type="button"
                        onClick={() => void submitQuery()}
                        disabled={!question.trim() || isQuerying}
                        className="inline-flex h-11 items-center gap-2 rounded-2xl border border-cyan-500/30 bg-cyan-500 px-5 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20 transition hover:-translate-y-0.5 hover:bg-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-400/45 disabled:translate-y-0 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none dark:disabled:border-white/10 dark:disabled:bg-white/5"
                      >
                        {isQuerying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                        Submit
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="min-h-[420px] p-5">
                <AnimatePresence mode="wait">
                  {result.state === "idle" ? (
                    <motion.div
                      key="idle"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="grid min-h-[360px] place-items-center rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 text-center dark:border-white/10 dark:bg-white/[0.03]"
                    >
                      <div className="max-w-md px-6">
                        <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-700 dark:text-cyan-200">
                          <PanelRight className="h-6 w-6" />
                        </div>
                        <h3 className="text-xl font-semibold tracking-tight">One governed query flow</h3>
                        <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">
                          Answers, refusals, evidence, and risk statistics appear here after the backend evaluates the request.
                        </p>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="response"
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -12 }}
                      className="space-y-4"
                    >
                      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-white/10 dark:bg-[#0d1118]">
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">Question</p>
                        <p className="mt-2 text-base leading-7 text-slate-800 dark:text-slate-100">{result.question}</p>
                      </div>

                      <div
                        className={cn(
                          "rounded-2xl border p-5",
                          result.state === "answer" &&
                            "border-emerald-500/25 bg-emerald-500/10",
                          result.state === "refusal" &&
                            "border-rose-500/25 bg-rose-500/10",
                          result.state === "error" &&
                            "border-amber-500/25 bg-amber-500/10",
                          result.state === "processing" &&
                            "border-cyan-500/25 bg-cyan-500/10",
                        )}
                      >
                        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                          <div className="flex items-center gap-3">
                            <div className="grid h-11 w-11 place-items-center rounded-2xl border border-current/20 bg-white/40 dark:bg-black/10">
                              {result.state === "answer" ? (
                                <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-300" />
                              ) : result.state === "refusal" ? (
                                <LockKeyhole className="h-5 w-5 text-rose-600 dark:text-rose-300" />
                              ) : result.state === "error" ? (
                                <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-300" />
                              ) : (
                                <Loader2 className="h-5 w-5 animate-spin text-cyan-600 dark:text-cyan-300" />
                              )}
                            </div>
                            <div>
                              <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">
                                {result.state === "answer"
                                  ? "Answer"
                                  : result.state === "refusal"
                                    ? "Access denied / cannot verify"
                                    : result.state === "error"
                                      ? "Error"
                                      : "Processing"}
                              </p>
                              <h3 className="mt-1 text-lg font-semibold">{result.status}</h3>
                            </div>
                          </div>
                          <div className="rounded-full border border-slate-200/70 bg-white/60 px-3 py-1 text-xs font-medium text-slate-600 dark:border-white/10 dark:bg-white/10 dark:text-slate-300">
                            Risk: {formatRisk(result.risk)}
                          </div>
                        </div>

                        <div className="prose prose-slate max-w-none whitespace-pre-wrap text-[15px] leading-8 text-slate-800 dark:prose-invert dark:text-slate-100">
                          {result.answer || "Waiting for streamed response chunks..."}
                        </div>

                        {result.state === "refusal" ? (
                          <div className="mt-5 rounded-2xl border border-rose-500/20 bg-white/55 p-4 dark:bg-black/10">
                            <p className="text-xs uppercase tracking-[0.2em] text-rose-700 dark:text-rose-200">Refusal reason</p>
                            <p className="mt-2 font-medium">{formatLabel(result.refusalReason) || "Policy refusal"}</p>
                          </div>
                        ) : null}
                      </div>

                      <div className="grid gap-3 md:grid-cols-3">
                        <ScoreRing label="Trust score" value={result.trustScore} />
                        <ScoreRing label="Faithfulness" value={result.faithfulnessScore} tone="emerald" />
                        <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
                          <p className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Decision</p>
                          <p className="mt-3 text-lg font-semibold">{lastQueryStatus}</p>
                          <p className="mt-1 truncate text-sm text-slate-500 dark:text-slate-400">
                            {result.queryId ? `Query ${result.queryId.slice(0, 8)}` : "Awaiting query id"}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-4">
              <MetricCard
                icon={Database}
                label="Azure health"
                value={health.backend === "online" ? "Online" : health.backend === "checking" ? "Checking" : "Offline"}
                detail={apiBaseUrl ? "Configured via env" : "Missing NEXT_PUBLIC_API_URL"}
              />
              <MetricCard
                icon={FileSearch}
                label="Vector DB"
                value={health.chroma === "online" ? "Ready" : health.chroma === "checking" ? "Checking" : "Offline"}
                detail={health.vectorStats?.backend ?? "Chroma status"}
              />
              <MetricCard
                icon={BarChart3}
                label="Total chunks"
                value={health.vectorStats?.total_chunks?.toLocaleString() ?? "--"}
                detail={health.vectorStats?.collection_name ?? "Collection pending"}
              />
              <MetricCard
                icon={ShieldCheck}
                label="Data presence"
                value={health.documentsCount !== null ? `${health.documentsCount} docs` : "--"}
                detail={`Last query: ${lastQueryStatus}`}
              />
            </section>
          </section>

          <aside className="space-y-4">
            <section className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-xl shadow-slate-200/50 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.055] dark:shadow-black/20">
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">Sources</p>
                  <h2 className="mt-1 text-xl font-semibold tracking-tight">Evidence Used</h2>
                </div>
              </div>

              <div className="mb-4 grid grid-cols-2 gap-2">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-white/10 dark:bg-white/5">
                  <p className="text-xs text-slate-500 dark:text-slate-400">Sources</p>
                  <p className="mt-1 text-xl font-semibold">{evidence.length}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-white/10 dark:bg-white/5">
                  <p className="text-xs text-slate-500 dark:text-slate-400">Documents</p>
                  <p className="mt-1 text-xl font-semibold">{health.documentsCount ?? "--"}</p>
                </div>
              </div>

              <div className="max-h-[560px] space-y-3 overflow-y-auto pr-1 pretty-scrollbar">
                {evidence.length ? (
                  evidence.map((trace, index) => {
                    const fallback = result.sources[index] ?? "";
                    return (
                      <motion.div
                        key={`${trace.chunk_id ?? fallback}-${index}`}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.03 }}
                        className="rounded-2xl border border-slate-200/70 bg-slate-50 p-4 dark:border-white/10 dark:bg-white/[0.04]"
                      >
                        <div className="flex items-start gap-3">
                          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-700 dark:text-cyan-200">
                            <FileSearch className="h-4 w-4" />
                          </div>
                          <div className="min-w-0">
                            <p className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Evidence {index + 1}</p>
                            <h3 className="mt-1 break-words text-sm font-semibold leading-6">
                              {sourceName(trace, fallback, index)}
                            </h3>
                            <div className="mt-3 flex flex-wrap gap-2">
                              <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                                Page {trace.page_number ?? "N/A"}
                              </span>
                              <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                                {formatLabel(trace.clause_type)}
                              </span>
                              <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
                                Match {formatPercent(trace.similarity_score)}
                              </span>
                            </div>
                            {trace.authority ? (
                              <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">Authority: {trace.authority}</p>
                            ) : null}
                          </div>
                        </div>
                      </motion.div>
                    );
                  })
                ) : (
                  <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-center dark:border-white/10 dark:bg-white/[0.03]">
                    <FileSearch className="mx-auto h-6 w-6 text-slate-400" />
                    <p className="mt-3 text-sm font-medium">
                      {result.state === "idle" ? "Evidence will appear with the governed response." : "No releasable evidence returned for this decision."}
                    </p>
                    <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Source details stay attached to the query result when the backend releases them.</p>
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-xl shadow-slate-200/50 backdrop-blur-xl dark:border-white/10 dark:bg-white/[0.055] dark:shadow-black/20">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">System</p>
                  <h2 className="mt-1 text-lg font-semibold">Backend Status</h2>
                </div>
                <button
                  type="button"
                  onClick={() => void refreshStatus()}
                  disabled={isRefreshing}
                  className="inline-flex h-9 items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-sm font-medium text-slate-600 shadow-sm transition hover:-translate-y-0.5 hover:border-cyan-300 hover:text-cyan-700 focus:outline-none focus:ring-2 focus:ring-cyan-400/35 disabled:translate-y-0 disabled:cursor-wait disabled:opacity-60 dark:border-white/10 dark:bg-white/5 dark:text-slate-300 dark:hover:bg-white/10"
                  aria-label="Refresh backend status"
                  title="Refresh backend status"
                >
                  <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
                  Refresh
                </button>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-slate-500 dark:text-slate-400">Health</span>
                  <StatusPill state={health.backend} label={health.backend} />
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-slate-500 dark:text-slate-400">Vector DB</span>
                  <StatusPill state={health.chroma} label={health.chroma} />
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-slate-500 dark:text-slate-400">Data files</span>
                  <StatusPill state={health.documents} label={health.documents} />
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-white/10 dark:bg-white/5">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Storage target</p>
                  <p className="mt-2 break-all text-xs leading-5 text-slate-600 dark:text-slate-300">
                    {health.vectorStats?.storage_target ?? "Waiting for vector stats"}
                  </p>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Last checked: {health.checkedAt ?? "pending"}
                </p>
              </div>
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}
