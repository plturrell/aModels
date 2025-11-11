import {
  AgentFlowCatalogResponse,
  AgentFlowImportResponse,
} from "../../types/api/agentflow";

const LANGFLOW_API_BASE = (process.env.REACT_APP_LANGFLOW_API_BASE ?? "/api/v1").replace(
  /\/$/,
  "",
);

async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${LANGFLOW_API_BASE}${path}`, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message =
      (typeof detail.detail === "string" && detail.detail.length > 0
        ? detail.detail
        : `Request failed (${response.status})`);
    throw new Error(message);
  }

  if (response.status === 204) {
    return {} as T;
  }
  return response.json();
}

export async function fetchAgentFlowCatalog(
  page = 1,
  pageSize = 20,
): Promise<AgentFlowCatalogResponse> {
  const search = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  return apiRequest<AgentFlowCatalogResponse>(`/agentflow/catalog?${search.toString()}`);
}

export async function importAgentFlow(flowId: string): Promise<AgentFlowImportResponse> {
  return apiRequest<AgentFlowImportResponse>("/agentflow/import", {
    method: "POST",
    body: JSON.stringify({ flow_id: flowId }),
  });
}
