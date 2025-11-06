import { useMemo } from "react";

import { Panel } from "../../components/Panel";
import { useDocuments, type DocumentRecord } from "../../api/hooks";

import styles from "./DocumentsModule.module.css";

const formatter = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit"
});

const relativeFormatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

const getRelativeTime = (iso: string) => {
  const now = Date.now();
  const timestamp = Date.parse(iso);
  if (Number.isNaN(timestamp)) return "unknown";
  const diffMs = timestamp - now;
  const diffMinutes = Math.round(diffMs / (1000 * 60));
  if (Math.abs(diffMinutes) < 60) {
    return relativeFormatter.format(diffMinutes, "minute");
  }
  const diffHours = Math.round(diffMinutes / 60);
  if (Math.abs(diffHours) < 24) {
    return relativeFormatter.format(diffHours, "hour");
  }
  const diffDays = Math.round(diffHours / 24);
  return relativeFormatter.format(diffDays, "day");
};

const createdThisWeek = (documents: DocumentRecord[]) => {
  const now = new Date();
  const startOfWeek = new Date(now);
  startOfWeek.setDate(now.getDate() - now.getDay());
  startOfWeek.setHours(0, 0, 0, 0);
  return documents.filter((doc) => {
    const created = Date.parse(doc.created_at);
    return !Number.isNaN(created) && created >= startOfWeek.getTime();
  }).length;
};

export function DocumentsModule() {
  const { data, loading, error, refresh } = useDocuments();

  const documents = useMemo(() => {
    const items = data ?? [];
    return [...items].sort(
      (a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)
    );
  }, [data]);

  const totalDocuments = documents.length;
  const newThisWeek = createdThisWeek(documents);
  const mostRecent = documents[0];

  return (
    <div className={styles.documents}>
      <Panel
        title="Document Library"
        subtitle="Curate, relate, and explore the latest uploads"
        actions={
          <button type="button" className={styles.refreshButton} onClick={refresh} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        }
      >
        <div className={styles.hero}>
          <div className={styles.heroCopy}>
            <h1>Everything you ingest lands here.</h1>
            <p>
              Track source material, surface relationships, and keep tabs on the freshest additions
              to your knowledge graph.
            </p>
          </div>

          <div className={styles.metricsRow}>
            <div className={styles.metricCard}>
              <span>Total documents</span>
              <strong>{totalDocuments}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>Added this week</span>
              <strong>{newThisWeek}</strong>
            </div>
            <div className={styles.metricCard}>
              <span>Latest upload</span>
              <strong>{mostRecent ? formatter.format(new Date(mostRecent.created_at)) : "—"}</strong>
            </div>
          </div>

          {error ? (
            <div className={styles.errorBanner}>Unable to load documents: {error.message}</div>
          ) : null}

          {!totalDocuments && !loading ? (
            <div className={styles.emptyState}>
              <span className={styles.emptyBadge}>Awaiting first upload</span>
              <p>
                Once the ingestion pipeline lands a document, this space will highlight the latest
                additions along with their relationships.
              </p>
            </div>
          ) : null}
        </div>
      </Panel>

      <div className={styles.layout}>
        <Panel title="Ledger" subtitle="Recently captured documents">
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Description</th>
                  <th>Created</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {documents.slice(0, 12).map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <div className={styles.cellTitle}>{doc.name}</div>
                      <div className={styles.cellMeta}>{doc.id}</div>
                    </td>
                    <td>{doc.description ?? "—"}</td>
                    <td>
                      <span>{formatter.format(new Date(doc.created_at))}</span>
                      <small>{getRelativeTime(doc.created_at)}</small>
                    </td>
                    <td>{formatter.format(new Date(doc.updated_at))}</td>
                  </tr>
                ))}
                {!documents.length ? (
                  <tr>
                    <td colSpan={4} className={styles.emptyCell}>
                      {loading ? "Loading…" : "No documents yet."}
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Next steps" subtitle="Bring the library to life" dense>
          <ul className={styles.actionList}>
            <li>
              <span className={styles.actionTitle}>Upload via DMS API</span>
              <p>Use the `/documents` endpoint to push new files from pipelines or CLI scripts.</p>
            </li>
            <li>
              <span className={styles.actionTitle}>Annotate relationships</span>
              <p>Sync taxonomy edges into Neo4j so downstream agents can traverse them.</p>
            </li>
            <li>
              <span className={styles.actionTitle}>Enrich with embeddings</span>
              <p>Queue Celery jobs to generate pgvector embeddings for RAG-ready search.</p>
            </li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}
