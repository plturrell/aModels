import { useMemo } from "react";

import { Panel } from "../../components/Panel";
import { telemetryDefaults, telemetryRecordFields, telemetryConfigFields } from "../../data/telemetry";
import {
  useTelemetryStore,
  type TelemetryState,
  type InteractionMetric
} from "../../state/useTelemetryStore";

import styles from "./TelemetryModule.module.css";

const formatNumber = (value: number | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "–";

const formatLatency = (value: number) => `${value.toLocaleString()} ms`;

const formatRelativeTime = (timestamp: number) => {
  const diff = Date.now() - timestamp;
  if (diff < 45_000) return "just now";
  if (diff < 90_000) return "1 min ago";
  const minutes = Math.round(diff / 60_000);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
};

export function TelemetryModule() {
  const metrics = useTelemetryStore((state: TelemetryState) => state.metrics);
  const resetMetrics = useTelemetryStore((state: TelemetryState) => state.reset);

  const summary = useMemo(() => {
    if (!metrics.length) {
      return null;
    }

    const totalLatency = metrics.reduce<number>((sum, metric) => sum + metric.durationMs, 0);
    const maxLatency = metrics.reduce<number>(
      (max, metric) => Math.max(max, metric.durationMs),
      0
    );
    const totalPromptTokens = metrics.reduce<number>(
      (sum, metric) => sum + (metric.promptTokens ?? 0),
      0
    );
    const totalCompletionTokens = metrics.reduce<number>(
      (sum, metric) => sum + (metric.completionTokens ?? 0),
      0
    );
    const totalCitations = metrics.reduce<number>(
      (sum, metric) => sum + metric.citations,
      0
    );

    return {
      interactions: metrics.length,
      avgLatency: Math.round(totalLatency / metrics.length),
      maxLatency,
      avgPromptTokens: Math.round(totalPromptTokens / metrics.length),
      avgCompletionTokens: Math.round(totalCompletionTokens / metrics.length),
      avgCitations:
        metrics.length > 0 ? Number((totalCitations / metrics.length).toFixed(1)) : 0
    };
  }, [metrics]);

  const hasMetrics = metrics.length > 0;

  return (
    <div className={styles.telemetry}>
      <Panel title="Service Defaults" subtitle="Telemetry baseline in extract service">
        <dl className={styles.defaults}>
          <div>
            <dt>Library</dt>
            <dd>{telemetryDefaults.library ?? "layer4_extract"}</dd>
          </div>
          <div>
            <dt>Operation</dt>
            <dd>{telemetryDefaults.operation ?? "run_extract"}</dd>
          </div>
          <div>
            <dt>HTTP timeout</dt>
            <dd>{telemetryDefaults.httpTimeout ?? "45 * time.Second"}</dd>
          </div>
          <div>
            <dt>Dial timeout</dt>
            <dd>{telemetryDefaults.dialTimeout ?? "5 * time.Second"}</dd>
          </div>
          <div>
            <dt>Call timeout</dt>
            <dd>{telemetryDefaults.callTimeout ?? "3 * time.Second"}</dd>
          </div>
        </dl>
      </Panel>

      <Panel
        title="Session Pulse"
        subtitle={hasMetrics && summary ? "Live feel of the LocalAI assistant" : "No interactions recorded yet"}
        actions={
          hasMetrics ? (
            <button type="button" className={styles.resetButton} onClick={resetMetrics}>
              Clear history
            </button>
          ) : null
        }
      >
        {hasMetrics && summary ? (
          <div className={styles.pulseGrid}>
            <div className={styles.pulseCard}>
              <span className={styles.pulseLabel}>Average latency</span>
              <strong>{formatLatency(summary.avgLatency)}</strong>
              <small>Peak {formatLatency(summary.maxLatency)}</small>
            </div>
            <div className={styles.pulseCard}>
              <span className={styles.pulseLabel}>Sessions today</span>
              <strong>{summary.interactions}</strong>
              <small>{summary.avgCitations.toFixed(1)} citations per reply</small>
            </div>
            <div className={styles.pulseCard}>
              <span className={styles.pulseLabel}>Token blend</span>
              <strong>
                {summary.avgPromptTokens.toLocaleString()} /{" "}
                {summary.avgCompletionTokens.toLocaleString()}
              </strong>
              <small>prompt / completion</small>
            </div>
          </div>
        ) : (
          <p className={styles.metricsEmpty}>
            Chat with LocalAI to light up latency, token, and citation telemetry in real time.
          </p>
        )}
      </Panel>

      <Panel title="Latest Sessions" subtitle="Rolling 25 interactions">
        {hasMetrics ? (
          <div className={styles.tableScroll}>
            <table className={styles.metricsTable}>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Model</th>
                  <th>Latency</th>
                  <th>Prompt tokens</th>
                  <th>Completion tokens</th>
                  <th>Citations</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric: InteractionMetric) => (
                  <tr key={metric.id}>
                    <td>{formatRelativeTime(metric.timestamp)}</td>
                    <td>
                      <span className={styles.pill}>{metric.model}</span>
                    </td>
                    <td>{formatLatency(metric.durationMs)}</td>
                    <td>{formatNumber(metric.promptTokens)}</td>
                    <td>{formatNumber(metric.completionTokens)}</td>
                    <td>{metric.citations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className={styles.metricsEmpty}>
            Once you start chatting we will surface latency, token usage, and citation coverage for
            each exchange here.
          </p>
        )}
      </Panel>

      <Panel title="Schema Reference" subtitle="Telemetry structs at a glance" dense>
        <div className={styles.schemaGrid}>
          <div>
            <h3>telemetryConfig</h3>
            <ul className={styles.fieldList}>
              {telemetryConfigFields.map((field) => (
                <li key={field}>{field}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3>telemetryRecord</h3>
            <ul className={styles.fieldList}>
              {telemetryRecordFields.map((field) => (
                <li key={field}>{field}</li>
              ))}
            </ul>
          </div>
        </div>
      </Panel>
    </div>
  );
}
