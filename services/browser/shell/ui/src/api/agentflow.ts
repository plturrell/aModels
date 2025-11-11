import { useEffect, useMemo, useState } from "react";

const AGENTFLOW_BASE = import.meta.env.VITE_AGENTFLOW_API ?? "http://localhost:9001";

async function fetchAgentflow<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${AGENTFLOW_BASE}${path}`;
  
  try {
    const response = await fetch(url, {
      headers: { 
        Accept: "application/json",
        "Content-Type": "application/json",
        ...init?.headers
      },
      ...init
    });

    if (!response.ok) {
      let errorMessage = `AgentFlow request failed (${response.status})`;
      try {
        const errorText = await response.text();
        if (errorText) {
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorText;
          } catch {
            errorMessage = errorText;
          }
        }
      } catch {
        errorMessage = `HTTP ${response.status} ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const text = await response.text();
      if (!text.trim()) {
        return null as T;
      }
      return JSON.parse(text) as T;
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach AgentFlow service at ${url}. Check if the service is running and accessible.`);
      }
      throw error;
    }
    throw new Error(`Unexpected error: ${String(error)}`);
  }
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
