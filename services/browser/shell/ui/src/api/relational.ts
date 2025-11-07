import { API_BASE } from "./client";

export interface RelationalTable {
  id: string;
  title: string;
  status: "succeeded" | "failed" | "processing";
  processed_at: string;
  catalog_id?: string;
  local_ai_id?: string;
  search_id?: string;
  metadata?: Record<string, any>;
  intelligence?: RelationalTableIntelligence;
}

export interface RelationalProcessingRequest {
  request_id: string;
  query: string;
  status: "pending" | "processing" | "completed" | "failed" | "partial";
  created_at: string;
  started_at?: string;
  completed_at?: string;
  processing_time_ms?: number;
  statistics?: {
    documents_processed: number;
    documents_succeeded: number;
    documents_failed: number;
    steps_completed: number;
  };
  current_step?: string;
  completed_steps?: string[];
  total_steps?: number;
  progress_percent?: number;
  estimated_time_remaining_ms?: number;
  errors?: Array<{ message: string; code?: string }>;
  warnings?: string[];
}

export interface RelationalTableIntelligence {
  domain?: string;
  domain_confidence?: number;
  relationships?: RelationalRelationship[];
  learned_patterns?: RelationalPattern[];
  catalog_patterns?: Record<string, any>;
  training_patterns?: Record<string, any>;
  domain_patterns?: Record<string, any>;
  search_patterns?: Record<string, any>;
}

export interface RelationalRelationship {
  type: string;
  target_id: string;
  target_title: string;
  strength: number;
}

export interface RelationalPattern {
  type: string;
  description: string;
  metadata?: Record<string, any>;
}

export interface RelationalRequestIntelligence {
  domains?: string[];
  total_relationships?: number;
  total_patterns?: number;
  knowledge_graph_nodes?: number;
  knowledge_graph_edges?: number;
  workflow_processed?: boolean;
  summary?: string;
}

export interface RelationalRequestHistory {
  requests: Array<{
    request_id: string;
    query: string;
    status: string;
    created_at: string;
    completed_at?: string;
    document_count: number;
  }>;
  total: number;
  limit: number;
  offset: number;
}

export interface RelationalSearchQuery {
  query: string;
  request_id?: string;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface RelationalSearchResults {
  query: string;
  results: Array<{
    document_id: string;
    title: string;
    score: number;
    content: string;
  }>;
  count: number;
}

/**
 * Get relational processing status for a request
 */
export async function getRelationalProcessingStatus(requestId: string): Promise<RelationalProcessingRequest> {
  const url = `${API_BASE}/api/relational/status/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch relational status: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get relational processing results for a request
 */
export async function getRelationalProcessingResults(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  statistics?: any;
  documents: RelationalTable[];
  results?: any;
  intelligence?: RelationalRequestIntelligence;
}> {
  const url = `${API_BASE}/api/relational/results/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch relational results: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get relational intelligence data for a request
 */
export async function getRelationalIntelligence(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  intelligence?: RelationalRequestIntelligence;
  documents: Array<{
    id: string;
    title: string;
    intelligence?: RelationalTableIntelligence;
  }>;
}> {
  const url = `${API_BASE}/api/relational/results/${requestId}/intelligence`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch relational intelligence: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get relational request history
 */
export async function getRelationalRequestHistory(params?: {
  limit?: number;
  offset?: number;
  status?: string;
  table?: string;
}): Promise<RelationalRequestHistory> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.set("limit", params.limit.toString());
  if (params?.offset) queryParams.set("offset", params.offset.toString());
  if (params?.status) queryParams.set("status", params.status);
  if (params?.table) queryParams.set("table", params.table);

  const url = `${API_BASE}/api/relational/history?${queryParams.toString()}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch relational history: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Search relational indexed tables
 */
export async function searchRelationalTables(query: RelationalSearchQuery): Promise<RelationalSearchResults> {
  const url = `${API_BASE}/api/relational/search`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(query)
  });
  if (!response.ok) {
    throw new Error(`Failed to search relational tables: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Process relational tables
 */
export async function processRelationalTables(params: {
  table?: string;
  tables?: string[];
  schema?: string;
  database_url?: string;
  database_type?: string;
  async?: boolean;
  webhook_url?: string;
  config?: Record<string, any>;
}): Promise<{
  status: string;
  request_id: string;
  message?: string;
  status_url?: string;
  results_url?: string;
  intelligence_url?: string;
}> {
  const url = `${API_BASE}/api/relational/process`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(params)
  });
  if (!response.ok) {
    throw new Error(`Failed to process relational tables: ${response.statusText}`);
  }
  return response.json();
}

