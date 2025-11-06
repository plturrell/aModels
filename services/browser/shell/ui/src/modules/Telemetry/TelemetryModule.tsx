import { useMemo } from "react";

import { Panel } from "../../components/Panel";
import { telemetryDefaults, telemetryRecordFields, telemetryConfigFields } from "../../data/telemetry";
import { useTelemetryStore, type TelemetryState, type InteractionMetric } from "../../state/useTelemetryStore";

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
    if (!metrics.length) return null;

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
      <Panel title="Extract Service Defaults" subtitle="services/extract/main.go">
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

      <Panel title="Telemetry Config Schema" subtitle="telemetryConfig struct">
        <ul className={styles.fieldList}>
          {telemetryConfigFields.map((field) => (
            <li key={field}>{field}</li>
          ))}
        </ul>
      </Panel>

      <Panel title="Telemetry Record Fields" subtitle="telemetryRecord struct">
        <ul className={styles.fieldList}>
          {telemetryRecordFields.map((field) => (
            <li key={field}>{field}</li>
          ))}
        </ul>
      </Panel>

      <Panel
        title="LocalAI Interaction Metrics"
        subtitle="Client-side telemetry (rolling 25)"
        actions={
          hasMetrics ? (
            <button type="button" className={styles.resetButton} onClick={resetMetrics}>
              Clear
            </button>
          ) : null
        }
      >
        <div className={styles.metricsPanel}>
          {hasMetrics && summary ? (
            <>
              <div className={styles.metricsSummary}>
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Interactions</span>
                  <strong>{summary.interactions}</strong>
                </div>
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Avg Latency</span>
                  <strong>{formatLatency(summary.avgLatency)}</strong>
                </div>
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Peak Latency</span>
                  <strong>{formatLatency(summary.maxLatency)}</strong>
                </div>
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Avg Tokens</span>
                  <strong>
                    {summary.avgPromptTokens.toLocaleString()} /{" "}
                    {summary.avgCompletionTokens.toLocaleString()}
                  </strong>
                  <span className={styles.summaryHint}>prompt / completion</span>
                </div>
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Avg Citations</span>
                  <strong>{summary.avgCitations.toFixed(1)}</strong>
                </div>
              </div>

              <div className={styles.tableScroll}>
                <table className={styles.metricsTable}>
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Model</th>
                      <th>Latency</th>
                      <th>Prompt&nbsp;tokens</th>
                      <th>Completion&nbsp;tokens</th>
                      <th>Citations</th>
                      <th>Prompt&nbsp;chars</th>
                      <th>Completion&nbsp;chars</th>
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
                        <td>{formatNumber(metric.promptChars)}</td>
                        <td>{formatNumber(metric.completionChars)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className={styles.metricsEmpty}>
              Run a conversation in the LocalAI module to populate latency, token, and citation
              metrics.
            </p>
          )}
        </div>
      </Panel>
    </div>
  );
}
