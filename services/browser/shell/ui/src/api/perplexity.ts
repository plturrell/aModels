import { PERPLEXITY_API_BASE } from "./client";
import type { GraphData } from "../types/graph";

export interface ProcessingRequest {
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
  document_ids?: string[];
  documents?: ProcessedDocument[];
  current_step?: string;
  completed_steps?: string[];
  total_steps?: number;
  progress_percent?: number;
  estimated_time_remaining_ms?: number;
  errors?: Array<{ message: string; code?: string }>;
  warnings?: string[];
  intelligence?: RequestIntelligence;
}

export interface ProcessedDocument {
  id: string;
  title?: string;
  status: "success" | "failed";
  processed_at: string;
  catalog_id?: string;
  training_task_id?: string;
  localai_id?: string;
  search_id?: string;
  error?: string;
  metadata?: Record<string, unknown>;
  intelligence?: DocumentIntelligence;
}

export interface DocumentIntelligence {
  domain?: string;
  domain_confidence?: number;
  knowledge_graph?: GraphData;
  workflow_results?: Record<string, unknown>;
  relationships?: Relationship[];
  learned_patterns?: Pattern[];
  catalog_patterns?: Record<string, unknown>;
  training_patterns?: Record<string, unknown>;
  domain_patterns?: Record<string, unknown>;
  search_patterns?: Record<string, unknown>;
  metadata_enrichment?: Record<string, unknown>;
}

export interface Relationship {
  source: string;
  target: string;
  type: string;
  strength?: number;
}

export interface Pattern {
  type: string;
  name: string;
  confidence?: number;
}

export interface RequestIntelligence {
  domains?: string[];
  total_relationships?: number;
  total_patterns?: number;
  knowledge_graph_nodes?: number;
  knowledge_graph_edges?: number;
  workflow_processed?: boolean;
  summary?: Record<string, unknown>;
  graph_data?: GraphData;
}

export interface RequestHistory {
  requests: ProcessingRequest[];
  total: number;
  limit: number;
  offset: number;
}

export interface SearchQuery {
  query: string;
  request_id?: string;
  top_k?: number;
  filters?: Record<string, unknown>;
}

export interface SearchResults {
  query: string;
  results: Array<Record<string, unknown>>;
  count: number;
}

/**
 * Get processing status for a request
 */
export async function getProcessingStatus(requestId: string): Promise<ProcessingRequest> {
  const url = `${PERPLEXITY_API_BASE}/api/perplexity/status/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get processing results for a request
 */
export async function getProcessingResults(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  documents: ProcessedDocument[];
  intelligence?: RequestIntelligence;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/perplexity/results/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch results: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get intelligence data for a request
 */
export async function getIntelligence(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  intelligence: RequestIntelligence;
  documents: Array<{
    id: string;
    title?: string;
    intelligence?: DocumentIntelligence;
  }>;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/perplexity/results/${requestId}/intelligence`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch intelligence: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get request history
 */
export async function getRequestHistory(options?: {
  limit?: number;
  offset?: number;
  status?: string;
  query?: string;
}): Promise<RequestHistory> {
  const params = new URLSearchParams();
  if (options?.limit) params.append("limit", options.limit.toString());
  if (options?.offset) params.append("offset", options.offset.toString());
  if (options?.status) params.append("status", options.status);
  if (options?.query) params.append("query", options.query);
  
  const queryString = params.toString();
  const url = `${PERPLEXITY_API_BASE}/api/perplexity/history${queryString ? `?${queryString}` : ""}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch history: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Search indexed documents
 */
export async function searchDocuments(request: SearchQuery): Promise<SearchResults> {
  const url = `${PERPLEXITY_API_BASE}/api/perplexity/search`;
  const response = await fetch(url, {
    method: "POST",
    body: JSON.stringify(request),
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    }
  });
  if (!response.ok) {
    throw new Error(`Failed to search: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Process documents (submit new request)
 */
export async function processDocuments(request: {
  query: string;
  model?: string;
  limit?: number;
  include_images?: boolean;
  async?: boolean;
  webhook_url?: string;
  config?: Record<string, unknown>;
}): Promise<{
  request_id: string;
  status: string;
  message?: string;
  status_url?: string;
  results_url?: string;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/perplexity/process`;
  const response = await fetch(url, {
    method: "POST",
    body: JSON.stringify(request),
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    }
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to process: ${errorText || response.statusText}`);
  }
  return response.json();
}

