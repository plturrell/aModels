import { useMemo, useState } from "react";

import { Panel } from "../../components/Panel";
import { useFlows, runFlow, type FlowInfo, type FlowRunResponse } from "../../api/agentflow";

import styles from "./FlowsModule.module.css";

const formatter = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit"
});

const relativeFormatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

const relativeTime = (iso?: string | null) => {
  if (!iso) return "—";
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return "—";
  const diffMs = ts - Date.now();
  const minutes = Math.round(diffMs / (1000 * 60));
  if (Math.abs(minutes) < 60) return relativeFormatter.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return relativeFormatter.format(hours, "hour");
  const days = Math.round(hours / 24);
  return relativeFormatter.format(days, "day");
};

const truncate = (value?: string | null, length = 140) => {
  if (!value) return null;
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
};

export function FlowsModule() {
  const { data, loading, error, refresh } = useFlows();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState<string>("Hello AgentFlow");
  const [sending, setSending] = useState<boolean>(false);
  const [runOutput, setRunOutput] = useState<FlowRunResponse | null>(null);
  const [runError, setRunError] = useState<Error | null>(null);

  const flows = useMemo(() => data ?? [], [data]);
  const totalFlows = flows.length;
  const syncedFlows = flows.filter((flow) => Boolean(flow.remote_id)).length;
  const pendingSync = totalFlows - syncedFlows;
  const recentlyUpdated = flows.filter((flow) => {
    if (!flow.updated_at) return false;
    const updated = Date.parse(flow.updated_at);
    if (Number.isNaN(updated)) return false;
    const diff = Date.now() - updated;
    return diff <= 1000 * 60 * 60 * 24 * 7;
  }).length;

  const selectedFlow: FlowInfo | undefined = selectedId
    ? flows.find((flow) => flow.local_id === selectedId)
    : undefined;

  const handleSelect = (flow: FlowInfo) => {
    setSelectedId(flow.local_id);
    setRunOutput(null);
    setRunError(null);
  };

  const handleRun = async () => {
    if (!selectedFlow) return;
    setSending(true);
    setRunError(null);
    try {
      const response = await runFlow(selectedFlow.local_id, {
        input_value: inputValue,
        ensure: true
      });
      setRunOutput(response);
    } catch (err) {
      setRunOutput(null);
      setRunError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className={styles.flows}>
      <Panel
        title="AgentFlow"
        subtitle="Manage, sync, and execute LangFlow pipelines"
        actions={
          <button type="button" className={styles.refreshButton} onClick={refresh} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        }
      >
        <div className={styles.hero}>
          <div className={styles.heroCopy}>
            <h1>Curate your orchestration lineup.</h1>
            <p>
              Preview local flows, monitor LangFlow sync status, and execute test runs without leaving
              the browser shell.
            </p>
          </div>

          <div className={styles.metricsRow}>
            <div className={styles.metricCard}>
              <span>Total flows</span>
              <strong>{totalFlows}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>Synced to LangFlow</span>
              <strong>{syncedFlows}</strong>
              <small>{pendingSync} pending sync</small>
            </div>
            <div className={styles.metricCard}>
              <span>Updated this week</span>
              <strong>{recentlyUpdated}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>Selected flow</span>
              <strong>{selectedFlow?.name ?? "—"}</strong>
              <small>{selectedFlow ? selectedFlow.local_id : "Choose from the ledger"}</small>
            </div>
          </div>

          {error ? (
            <div className={styles.errorBanner}>Unable to load flows: {error.message}</div>
          ) : null}
        </div>
      </Panel>

      <div className={styles.layout}>
        <Panel title="Flow ledger" subtitle="Local specs and sync status">
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Project</th>
                  <th>Folder</th>
                  <th>Status</th>
                  <th>Updated</th>
                  <th>Synced</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {flows.map((flow) => (
                  <tr
                    key={flow.local_id}
                    className={flow.local_id === selectedId ? styles.selectedRow : undefined}
                  >
                    <td>
                      <div className={styles.cellTitle}>{flow.name ?? flow.local_id}</div>
                      <div className={styles.cellMeta}>{truncate(flow.description)}</div>
                    </td>
                    <td>{flow.project_id ?? "—"}</td>
                    <td>{flow.folder_path ?? "—"}</td>
                    <td>
                      {flow.remote_id ? (
                        <span className={styles.syncedPill}>Synced</span>
                      ) : (
                        <span className={styles.pendingPill}>Local only</span>
                      )}
                    </td>
                    <td>
                      {flow.updated_at ? formatter.format(new Date(flow.updated_at)) : "—"}
                      <small>{relativeTime(flow.updated_at)}</small>
                    </td>
                    <td>
                      {flow.synced_at ? formatter.format(new Date(flow.synced_at)) : "—"}
                      <small>{relativeTime(flow.synced_at)}</small>
                    </td>
                    <td className={styles.actionsCell}>
                      <button
                        type="button"
                        className={styles.secondaryButton}
                        onClick={() => handleSelect(flow)}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
                {!flows.length && !loading ? (
                  <tr>
                    <td colSpan={7} className={styles.emptyCell}>
                      No flows discovered in the AgentFlow catalog.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Execution studio" subtitle="Run the selected flow">
          {selectedFlow ? (
            <div className={styles.runPanel}>
              <div className={styles.runMeta}>
                <div>
                  <span className={styles.metaLabel}>Local ID</span>
                  <p>{selectedFlow.local_id}</p>
                </div>
                <div>
                  <span className={styles.metaLabel}>Remote ID</span>
                  <p>{selectedFlow.remote_id ?? "Pending sync"}</p>
                </div>
              </div>
              <label className={styles.inputLabel} htmlFor="flow-input">
                Input value
              </label>
              <textarea
                id="flow-input"
                className={styles.promptInput}
                value={inputValue}
                rows={4}
                onChange={(event) => setInputValue(event.target.value)}
                placeholder="Provide the primary input value for this flow run"
              />
              <div className={styles.runActions}>
                <button
                  type="button"
                  className={styles.primaryButton}
                  onClick={handleRun}
                  disabled={!inputValue.trim() || sending}
                >
                  {sending ? "Running…" : "Run flow"}
                </button>
              </div>

              {runError ? <div className={styles.errorBanner}>Run failed: {runError.message}</div> : null}

              {runOutput ? (
                <div className={styles.resultBlock}>
                  <span className={styles.metaLabel}>Result payload</span>
                  <pre>{JSON.stringify(runOutput.result, null, 2)}</pre>
                  {runOutput.deepagents_analysis ? (
                    <>
                      <span className={styles.metaLabel}>DeepAgents analysis</span>
                      <pre>{JSON.stringify(runOutput.deepagents_analysis, null, 2)}</pre>
                    </>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : (
            <div className={styles.placeholder}>
              <p>Select a flow from the ledger to run it against LangFlow.</p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
