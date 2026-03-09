"use client";

import { useEffect, useMemo, useState } from "react";

type Run = {
  id: string;
  mode: "STATIC" | "ADAPTIVE";
  status: "RUNNING" | "COMPLETED" | "FAILED";
  createdAt: string;
  startedAt: string | null;
  endedAt: string | null;
  error: string | null;
  summaryJson: string | null;
  _count?: { recommendations: number };
};

type Recommendation = {
  id: string;
  ticker: string;
  decision: string;
  date: string | null;
  close: number | null;
  expectedReturn: number | null;
  probPositive: number | null;
  decisionConfidence: number | null;
  uncertainty: number | null;
  portfolioWeight: number | null;
  auditJson: string | null;
};

type RunDetail = Run & { recommendations: Recommendation[] };
type AuditPayload = Record<string, unknown>;
type RunModeChoice = "auto" | "static" | "adaptive";

function formatNumber(value: number | null, decimals = 3): string {
  if (value === null || Number.isNaN(value)) return "-";
  return value.toFixed(decimals);
}

function formatPercent(value: number | null, decimals = 1): string {
  if (value === null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(decimals)}%`;
}

function parseAuditJson(auditJson: string | null): AuditPayload | null {
  if (!auditJson) return null;
  try {
    return JSON.parse(auditJson) as AuditPayload;
  } catch {
    return null;
  }
}

function decisionRank(decision: string): number {
  if (decision === "Buy") return 0;
  if (decision === "Sell") return 1;
  return 2;
}

export default function HomePage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [selectedRecId, setSelectedRecId] = useState<string | null>(null);
  const [advancedControls, setAdvancedControls] = useState(false);
  const [modeChoice, setModeChoice] = useState<RunModeChoice>("auto");
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadRuns() {
    const res = await fetch("/api/runs", { cache: "no-store" });
    const data = await res.json();
    setRuns(data.runs ?? []);
  }

  async function loadRun(id: string) {
    const res = await fetch(`/api/runs/${id}`, { cache: "no-store" });
    const data = await res.json();
    if (data.run) {
      setSelected(data.run);
      setSelectedRecId(data.run.recommendations?.[0]?.id ?? null);
    }
  }

  async function postRun(mode: "static" | "adaptive") {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, data };
  }

  async function hydrateRunState(runId: string | null) {
    await loadRuns();
    if (runId) {
      await loadRun(runId);
    }
  }

  async function startAnalysis() {
    setBusy(true);
    setError(null);
    setInfo(null);
    try {
      const effectiveMode: RunModeChoice = advancedControls ? modeChoice : "auto";
      if (effectiveMode === "auto") {
        const adaptive = await postRun("adaptive");
        if (adaptive.ok) {
          setInfo("Auto mode executed with adaptive engine.");
          await hydrateRunState(adaptive.data?.run?.id ?? null);
          return;
        }

        const staticFallback = await postRun("static");
        if (staticFallback.ok) {
          setInfo("Adaptive failed. Auto mode fell back to static engine.");
          await hydrateRunState(staticFallback.data?.run?.id ?? null);
          return;
        }

        setError(
          [
            adaptive.data?.details ?? adaptive.data?.error ?? "Adaptive failed",
            staticFallback.data?.details ?? staticFallback.data?.error ?? "Static fallback failed",
          ].join(" | "),
        );
        await loadRuns();
        return;
      }

      const single = await postRun(effectiveMode);
      if (!single.ok) {
        setError(single.data?.details ?? single.data?.error ?? "Run failed");
      } else {
        setInfo(`Run completed with ${effectiveMode.toUpperCase()} engine.`);
      }
      await hydrateRunState(single.data?.run?.id ?? null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void loadRuns();
  }, []);

  const summary = useMemo(() => {
    if (!selected?.summaryJson) return null;
    try {
      return JSON.parse(selected.summaryJson) as Record<string, number>;
    } catch {
      return null;
    }
  }, [selected]);

  const sortedRecommendations = useMemo(() => {
    if (!selected) return [];
    return [...selected.recommendations].sort((a, b) => {
      const rankDiff = decisionRank(a.decision) - decisionRank(b.decision);
      if (rankDiff !== 0) return rankDiff;
      return (b.portfolioWeight ?? 0) - (a.portfolioWeight ?? 0);
    });
  }, [selected]);

  const selectedRecommendation = useMemo(() => {
    if (!sortedRecommendations.length) return null;
    if (!selectedRecId) return sortedRecommendations[0] ?? null;
    return sortedRecommendations.find((r) => r.id === selectedRecId) ?? sortedRecommendations[0];
  }, [sortedRecommendations, selectedRecId]);

  const selectedAudit = useMemo(() => {
    return parseAuditJson(selectedRecommendation?.auditJson ?? null);
  }, [selectedRecommendation]);

  const auditSections = useMemo(() => {
    if (!selectedAudit) return [];
    const sectionNames = ["thresholds", "checks", "metrics", "signals", "regime", "risk"] as const;
    const sections: Array<{ name: string; value: Record<string, unknown> }> = [];
    for (const name of sectionNames) {
      const value = selectedAudit[name];
      if (!value || typeof value !== "object") continue;
      sections.push({ name, value: value as Record<string, unknown> });
    }
    return sections;
  }, [selectedAudit]);

  const summaryStats = useMemo(() => {
    if (!selected) return null;
    const total = summary?.recommendations_total ?? selected.recommendations.length;
    const buyCount = summary?.buy_count ?? selected.recommendations.filter((r) => r.decision === "Buy").length;
    const sellCount = summary?.sell_count ?? selected.recommendations.filter((r) => r.decision === "Sell").length;
    const holdCount = summary?.hold_count ?? selected.recommendations.filter((r) => r.decision === "Hold").length;
    const actionable = selected.recommendations.filter(
      (r) => r.decision === "Buy" && (r.portfolioWeight ?? 0) > 0,
    ).length;
    const allocated = selected.recommendations.reduce((sum, r) => sum + (r.portfolioWeight ?? 0), 0);
    return { total, buyCount, sellCount, holdCount, actionable, allocated };
  }, [selected, summary]);

  const healthLabel = useMemo(() => {
    if (!selected) return "No run selected";
    if (selected.status === "FAILED") return "Run failed";
    if ((summaryStats?.total ?? 0) === 0) return "No opportunities";
    const missingAudit = selected.recommendations.some((r) => !r.auditJson);
    if (missingAudit) return "Partial audit";
    return "Healthy";
  }, [selected, summaryStats]);

  const healthClass = useMemo(() => {
    if (!selected) return "health-neutral";
    if (selected.status === "FAILED") return "health-bad";
    if ((summaryStats?.total ?? 0) === 0) return "health-warn";
    return "health-good";
  }, [selected, summaryStats]);

  return (
    <main className="page">
      <section className="card hero-card">
        <h1>ROI Decision Console</h1>
        <p>
          Run equity analysis, review actionable recommendations, and inspect the full rationale behind every
          decision.
        </p>
        <div className="workflow">
          <span>1. Run Analysis</span>
          <span>2. Review Recommendations</span>
          <span>3. Inspect Decision Audit</span>
        </div>
      </section>

      <section className="card">
        <div className="section-head">
          <h2>Run Analysis</h2>
          <button disabled={busy} onClick={() => loadRuns()}>Refresh</button>
        </div>
        <div className="controls">
          <button className="primary" disabled={busy} onClick={() => startAnalysis()}>
            {busy ? "Running..." : "Run Analysis"}
          </button>
          <label className="inline-toggle">
            <input
              type="checkbox"
              checked={advancedControls}
              onChange={(event) => setAdvancedControls(event.target.checked)}
            />
            Advanced controls
          </label>
        </div>
        {advancedControls ? (
          <div className="advanced-panel">
            <label htmlFor="mode-select">Engine mode</label>
            <select
              id="mode-select"
              value={modeChoice}
              onChange={(event) => setModeChoice(event.target.value as RunModeChoice)}
            >
              <option value="auto">Auto (recommended)</option>
              <option value="adaptive">Adaptive</option>
              <option value="static">Static</option>
            </select>
            <p>Auto tries adaptive first and falls back to static if execution fails.</p>
          </div>
        ) : null}
        {info ? <p className="info-text">{info}</p> : null}
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      {selected && summaryStats ? (
        <section className="card summary-strip">
          <div className="summary-item">
            <span className="summary-label">Total Ideas</span>
            <span className="summary-value">{summaryStats.total}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Actionable Buys</span>
            <span className="summary-value">{summaryStats.actionable}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Allocated Capital</span>
            <span className="summary-value">{formatPercent(summaryStats.allocated, 1)}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Data Health</span>
            <span className={`health-pill ${healthClass}`}>{healthLabel}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Run Status</span>
            <span className={`status ${selected.status}`}>{selected.status}</span>
          </div>
        </section>
      ) : null}

      <section className="grid">
        <div className="card">
          <h2>Recent Runs</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Created</th>
                <th>Mode</th>
                <th>Status</th>
                <th>Rows</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.id}
                  className={`clickable-row ${selected?.id === run.id ? "selected-row" : ""}`}
                  onClick={() => loadRun(run.id)}
                >
                  <td>{new Date(run.createdAt).toLocaleString()}</td>
                  <td>{run.mode}</td>
                  <td><span className={`status ${run.status}`}>{run.status}</span></td>
                  <td>{run._count?.recommendations ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>Run Details</h2>
          {!selected ? <p>Select a run to inspect output.</p> : null}
          {selected ? (
            <>
              <p>
                {selected.mode} | <span className={`status ${selected.status}`}>{selected.status}</span>
              </p>
              {summaryStats ? (
                <p style={{ marginTop: 10 }}>
                  total={summaryStats.total}, buy={summaryStats.buyCount}, sell={summaryStats.sellCount}, hold=
                  {summaryStats.holdCount}
                </p>
              ) : null}

              {sortedRecommendations.length === 0 ? (
                <div className="empty-state">
                  <strong>No recommendations in this run.</strong>
                  <p>
                    This usually means filters or risk checks blocked all candidates. Open a different run or run
                    analysis again with a less restrictive mode.
                  </p>
                </div>
              ) : null}

              {sortedRecommendations.length > 0 ? (
                <>
                  <div className="recommendation-cards">
                    {sortedRecommendations.slice(0, 8).map((rec) => {
                      const audit = parseAuditJson(rec.auditJson);
                      const rationale = typeof audit?.summary === "string" ? audit.summary : "No rationale.";
                      return (
                        <button
                          key={rec.id}
                          type="button"
                          className={`rec-card ${selectedRecId === rec.id ? "selected-rec-card" : ""}`}
                          onClick={() => setSelectedRecId(rec.id)}
                        >
                          <div className="rec-top">
                            <strong>{rec.ticker}</strong>
                            <span className={`decision-pill ${rec.decision}`}>{rec.decision}</span>
                          </div>
                          <div className="rec-metrics">
                            <span>E[r] {formatPercent(rec.expectedReturn, 2)}</span>
                            <span>P(up) {formatPercent(rec.probPositive, 1)}</span>
                            <span>Weight {formatPercent(rec.portfolioWeight, 1)}</span>
                          </div>
                          <p>{rationale}</p>
                        </button>
                      );
                    })}
                  </div>

                  <table className="table" style={{ marginTop: 14 }}>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Decision</th>
                        <th>E[r]</th>
                        <th>P(up)</th>
                        <th>Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedRecommendations.slice(0, 50).map((rec) => (
                        <tr
                          key={rec.id}
                          className={`clickable-row ${selectedRecId === rec.id ? "selected-row" : ""}`}
                          onClick={() => setSelectedRecId(rec.id)}
                        >
                          <td>{rec.ticker}</td>
                          <td>{rec.decision}</td>
                          <td>{rec.expectedReturn?.toFixed(4) ?? "-"}</td>
                          <td>{rec.probPositive?.toFixed(3) ?? "-"}</td>
                          <td>{rec.portfolioWeight?.toFixed(3) ?? "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : null}

              {selectedRecommendation ? (
                <section className="audit-panel">
                  <h3>
                    Decision Audit: {selectedRecommendation.ticker} ({selectedRecommendation.decision})
                  </h3>
                  <p>
                    {typeof selectedAudit?.summary === "string"
                      ? selectedAudit.summary
                      : "No structured rationale available for this row."}
                  </p>

                  <div className="audit-kv">
                    <div><strong>Date:</strong> {selectedRecommendation.date ? new Date(selectedRecommendation.date).toLocaleString() : "-"}</div>
                    <div><strong>Close:</strong> {formatNumber(selectedRecommendation.close, 2)}</div>
                    <div><strong>E[r]:</strong> {formatPercent(selectedRecommendation.expectedReturn, 2)}</div>
                    <div><strong>P(up):</strong> {formatPercent(selectedRecommendation.probPositive, 1)}</div>
                    <div><strong>Confidence:</strong> {formatPercent(selectedRecommendation.decisionConfidence, 1)}</div>
                    <div><strong>Uncertainty:</strong> {formatPercent(selectedRecommendation.uncertainty, 1)}</div>
                    <div><strong>Weight:</strong> {formatPercent(selectedRecommendation.portfolioWeight, 1)}</div>
                  </div>

                  {auditSections.map((section) => (
                    <details key={section.name} open={section.name === "checks" || section.name === "metrics"}>
                      <summary>{section.name}</summary>
                      <pre className="mono">{JSON.stringify(section.value, null, 2)}</pre>
                    </details>
                  ))}

                  <details>
                    <summary>raw_fields</summary>
                    <pre className="mono">{JSON.stringify(selectedAudit?.raw_fields ?? {}, null, 2)}</pre>
                  </details>
                </section>
              ) : null}
            </>
          ) : null}
        </div>
      </section>
    </main>
  );
}
