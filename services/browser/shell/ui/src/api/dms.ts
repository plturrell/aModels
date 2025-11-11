/**
 * DMS API client for Browser Shell
 */

import { PERPLEXITY_API_BASE } from "./client";
import type { GraphData } from "../types/graph";

export interface DMSProcessingRequest {
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
  document_ids: string[];
  documents?: DMSProcessedDocument[];
  current_step?: string;
  completed_steps?: string[];
  total_steps?: number;
  progress_percent?: number;
  estimated_time_remaining_ms?: number;
  errors?: Array<string | { message: string; code?: string }>;
  warnings?: string[];
  intelligence?: DMSRequestIntelligence;
}

export interface DMSProcessedDocument {
  id: string;
  title: string;
  status: string;
  processed_at: string;
  catalog_id?: string;
  local_ai_id?: string;
  search_id?: string;
  error?: string;
  intelligence?: DMSDocumentIntelligence;
}

export interface DMSDocumentIntelligence {
  domain?: string;
  domain_confidence?: number;
  knowledge_graph?: GraphData;
  workflow_results?: Record<string, unknown>;
  relationships?: DMSRelationship[];
  learned_patterns?: DMSPattern[];
  catalog_patterns?: Record<string, unknown>;
  training_patterns?: Record<string, unknown>;
  domain_patterns?: Record<string, unknown>;
  search_patterns?: Record<string, unknown>;
  metadata_enrichment?: Record<string, unknown>;
}

export interface DMSRequestIntelligence {
  domains: string[];
  total_relationships: number;
  total_patterns: number;
  knowledge_graph_nodes: number;
  knowledge_graph_edges: number;
  workflow_processed: boolean;
  summary?: string;
  graph_data?: GraphData;
}

export interface DMSRelationship {
  type: string;
  target_id: string;
  target_title?: string;
  strength: number;
  metadata?: Record<string, unknown>;
}

export interface DMSPattern {
  type: string;
  description: string;
  metadata?: Record<string, unknown>;
}

export interface DMSRequestHistory {
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

export interface DMSDocument {
  id: string;
  name: string;
  description?: string;
  storage_path: string;
  catalog_identifier?: string;
  extraction_summary?: string;
  created_at: string;
  updated_at: string;
}

/**
 * Get processing status for a DMS request
 */
export async function getDMSStatus(requestId: string): Promise<DMSProcessingRequest> {
  const url = `${PERPLEXITY_API_BASE}/api/dms/status/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get processing results for a DMS request
 */
export async function getDMSResults(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  statistics?: DMSProcessingRequest["statistics"];
  documents: DMSProcessedDocument[];
  results?: {
    catalog_url?: string;
    search_url?: string;
    export_url?: string;
  };
  intelligence?: DMSRequestIntelligence;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/dms/results/${requestId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch results: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get intelligence data for a DMS request
 */
export async function getDMSIntelligence(requestId: string): Promise<{
  request_id: string;
  query: string;
  status: string;
  intelligence?: DMSRequestIntelligence;
  documents: Array<{
    id: string;
    title: string;
    intelligence?: DMSDocumentIntelligence;
  }>;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/dms/results/${requestId}/intelligence`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch intelligence: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get DMS request history
 */
export async function getDMSHistory(options: {
  limit?: number;
  offset?: number;
  status?: string;
  document_id?: string;
} = {}): Promise<DMSRequestHistory> {
  const params = new URLSearchParams();
  if (options.limit) params.set("limit", options.limit.toString());
  if (options.offset) params.set("offset", options.offset.toString());
  if (options.status) params.set("status", options.status);
  if (options.document_id) params.set("document_id", options.document_id);

  const url = `${PERPLEXITY_API_BASE}/api/dms/history?${params.toString()}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch history: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Process DMS documents
 */
export async function processDMSDocuments(payload: {
  document_id?: string;
  document_ids?: string[];
  async?: boolean;
  webhook_url?: string;
  config?: Record<string, unknown>;
}): Promise<{
  status: string;
  request_id: string;
  status_url?: string;
  results_url?: string;
  message?: string;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/dms/process`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to process documents: ${error}`);
  }
  return response.json();
}

/**
 * List DMS documents
 */
export async function listDMSDocuments(options: {
  limit?: number;
  offset?: number;
} = {}): Promise<DMSDocument[]> {
  const params = new URLSearchParams();
  if (options.limit) params.set("limit", options.limit.toString());
  if (options.offset) params.set("offset", options.offset.toString());

  const dmsUrl = import.meta.env.VITE_DMS_URL || "http://localhost:8096";
  const url = `${dmsUrl}/documents?${params.toString()}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch documents: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get a specific DMS document
 */
export async function getDMSDocument(documentId: string): Promise<DMSDocument> {
  const dmsUrl = import.meta.env.VITE_DMS_URL || "http://localhost:8096";
  const url = `${dmsUrl}/documents/${documentId}`;
  const response = await fetch(url, {
    headers: { Accept: "application/json" }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch document: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Search DMS documents
 */
export async function searchDMSDocuments(query: {
  query: string;
  request_id?: string;
  top_k?: number;
  filters?: Record<string, unknown>;
}): Promise<{
  query: string;
  results: Array<Record<string, unknown>>;
  count: number;
}> {
  const url = `${PERPLEXITY_API_BASE}/api/dms/search`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(query)
  });
  if (!response.ok) {
    throw new Error(`Failed to search documents: ${response.statusText}`);
  }
  return response.json();
}

