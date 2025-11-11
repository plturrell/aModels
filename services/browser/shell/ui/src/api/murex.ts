import { API_BASE } from "./client";
import type { GraphData } from "../types/graph";

export interface MurexTrade {
  id: string;
  title: string;
  status: "succeeded" | "failed" | "processing";
  processed_at: string;
  catalog_id?: string;
  local_ai_id?: string;
  search_id?: string;
  metadata?: Record<string, any>;
  intelligence?: MurexTradeIntelligence;
}

export interface MurexProcessingRequest {
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

export interface MurexTradeIntelligence {
  domain?: string;
  domain_confidence?: number;
  relationships?: MurexRelationship[];
  learned_patterns?: MurexPattern[];
  graph_data?: GraphData;
}

export interface MurexRelationship {
  type: string;
  target_id: string;
  target_title: string;
  strength: number;
}

export interface MurexPattern {
  type: string;
  description: string;
  metadata?: Record<string, any>;
}

export interface MurexRequestIntelligence {
  domains?: string[];
  total_relationships?: number;
  total_patterns?: number;
  knowledge_graph_nodes?: number;
  knowledge_graph_edges?: number;
  workflow_processed?: boolean;
  summary?: string;
  graph_data?: GraphData;
}

export interface MurexRequestHistory {
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

/**
 * Get Murex processing status for a request
 */
export async function getMurexProcessingStatus(requestId: string): Promise<MurexProcessingRequest> {
  const url = `${API_BASE}/api/murex/status/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch Murex status: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get Murex processing results for a request
 */
export async function getMurexProcessingResults(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  statistics?: any;
  documents: MurexTrade[];
  results?: any;
  intelligence?: MurexRequestIntelligence;
}> {
  const url = `${API_BASE}/api/murex/results/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch Murex results: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get Murex intelligence data for a request
 */
export async function getMurexIntelligence(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  intelligence?: MurexRequestIntelligence;
  documents: Array<{
    id: string;
    title: string;
    intelligence?: MurexTradeIntelligence;
  }>;
}> {
  const url = `${API_BASE}/api/murex/results/${requestId}/intelligence`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch Murex intelligence: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get Murex request history
 */
export async function getMurexRequestHistory(params?: {
  limit?: number;
  offset?: number;
  status?: string;
  table?: string;
}): Promise<MurexRequestHistory> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.set("limit", params.limit.toString());
  if (params?.offset) queryParams.set("offset", params.offset.toString());
  if (params?.status) queryParams.set("status", params.status);
  if (params?.table) queryParams.set("table", params.table);

  const url = `${API_BASE}/api/murex/history?${queryParams.toString()}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch Murex history: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Process Murex trades
 */
export async function processMurexTrades(params: {
  table?: string;
  filters?: Record<string, any>;
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
  const url = `${API_BASE}/api/murex/process`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(params)
  });
  if (!response.ok) {
    throw new Error(`Failed to process Murex trades: ${response.statusText}`);
  }
  return response.json();
}

