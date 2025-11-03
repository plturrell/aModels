import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchAgentFlowCatalog,
  importAgentFlow,
} from "../../../controllers/API/agentflowAPI";
import { AgentFlowCatalogItem } from "../../../types/api/agentflow";

const PAGE_SIZE = 10;

const AgentFlowPage: React.FC = () => {
  const navigate = useNavigate();
  const [items, setItems] = useState<AgentFlowCatalogItem[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [importing, setImporting] = useState<string | null>(null);
  const [importMessage, setImportMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadCatalog() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchAgentFlowCatalog(page, PAGE_SIZE);
        if (!cancelled) {
          setItems(response.items);
          setTotal(response.total);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load AgentFlow catalog.");
          setItems([]);
          setTotal(0);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadCatalog();
    return () => {
      cancelled = true;
    };
  }, [page]);

  const handleImport = async (flowId: string) => {
    setImporting(flowId);
    setError(null);
    setImportMessage(null);
    try {
      const result = await importAgentFlow(flowId);
      const importedId = result.flow?.id;
      if (importedId) {
        navigate(`/flow/${importedId}/`);
        return;
      }
      setImportMessage("AgentFlow definition imported successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to import AgentFlow definition.");
    } finally {
      setImporting(null);
    }
  };

  const hasPrevious = page > 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const hasNext = page < totalPages;

  return (
    <div className="flex flex-col h-full bg-background text-foreground">
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold">AgentFlow Catalog</h1>
          <p className="mt-2 text-sm text-muted">
            Browse AgentFlow specifications and import them into this Langflow instance.
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {importMessage && (
          <div className="mb-4 rounded-md border border-green-300 bg-green-50 px-4 py-3 text-sm text-green-700">
            {importMessage}
          </div>
        )}

        <div className="rounded-md border border-border bg-card shadow-sm">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border text-sm">
              <thead className="bg-muted/40">
                <tr>
                  <th className="px-4 py-3 text-left font-medium uppercase tracking-wide text-xs">
                    Name
                  </th>
                  <th className="px-4 py-3 text-left font-medium uppercase tracking-wide text-xs">
                    Category
                  </th>
                  <th className="px-4 py-3 text-left font-medium uppercase tracking-wide text-xs">
                    Description
                  </th>
                  <th className="px-4 py-3 text-left font-medium uppercase tracking-wide text-xs">
                    Path
                  </th>
                  <th className="px-4 py-3 text-right font-medium uppercase tracking-wide text-xs">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-card">
                {loading ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-6 text-center text-muted">
                      Loading AgentFlow catalog…
                    </td>
                  </tr>
                ) : items.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-6 text-center text-muted">
                      No AgentFlow specifications were found.
                    </td>
                  </tr>
                ) : (
                  items.map((item) => (
                    <tr key={item.id} className="hover:bg-muted/20">
                      <td className="px-4 py-3 font-medium">{item.name}</td>
                      <td className="px-4 py-3">{item.category ?? "—"}</td>
                      <td className="px-4 py-3 text-muted">
                        {item.description ?? "No description provided."}
                      </td>
                      <td className="px-4 py-3 text-xs text-muted">{item.relative_path}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          onClick={() => handleImport(item.id)}
                          disabled={importing === item.id}
                          className="inline-flex items-center rounded-md border border-primary px-3 py-1 text-sm font-medium text-primary transition hover:bg-primary/10 disabled:opacity-60"
                        >
                          {importing === item.id ? "Importing…" : "Import"}
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between border-t border-border px-4 py-3 text-sm text-muted">
            <div>
              Page {page} of {totalPages}
            </div>
            <div className="space-x-2">
              <button
                type="button"
                onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                disabled={!hasPrevious || loading}
                className="rounded-md border border-border px-3 py-1 transition hover:bg-muted/40 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Previous
              </button>
              <button
                type="button"
                onClick={() => setPage((prev) => prev + 1)}
                disabled={!hasNext || loading}
                className="rounded-md border border-border px-3 py-1 transition hover:bg-muted/40 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentFlowPage;
