import { useEffect, useMemo, useState } from "react";

const AGENTFLOW_BASE = import.meta.env.VITE_AGENTFLOW_API ?? "/agentflow";

async function fetchAgentflow<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${AGENTFLOW_BASE}${path}`, {
    headers: { Accept: "application/json" },
    ...init
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`AgentFlow request failed (${response.status}): ${message}`);
  }
  return (await response.json()) as T;
}

export interface FlowInfo {
  local_id: string;
  remote_id?: string | null;
  name?: string | null;
  description?: string | null;
  project_id?: string | null;
  folder_path?: string | null;
  updated_at?: string | null;
  synced_at?: string | null;
}

export interface FlowRunPayload {
  input_value: string;
  session_id?: string;
  ensure?: boolean;
}

export interface FlowRunResponse {
  local_id: string;
  remote_id?: string;
  result: unknown;
  deepagents_analysis?: unknown;
}

export function useFlows() {
  const [data, setData] = useState<FlowInfo[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const refresh = useMemo(() => () => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchAgentflow<FlowInfo[]>("/flows")
      .then((value) => {
        if (cancelled) return;
        setData(value);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        setData(null);
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [reloadToken]);

  return { data, loading, error, refresh };
}

export async function runFlow(localId: string, payload: FlowRunPayload) {
  return fetchAgentflow<FlowRunResponse>(`/flows/${encodeURIComponent(localId)}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify({
      ...payload,
      ensure: payload.ensure ?? true
    })
  });
}
