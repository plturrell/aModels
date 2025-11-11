/**
 * Extract Service API Client
 * 
 * Client for interacting with extract service endpoints for entity extraction,
 * knowledge graph processing, OCR, and schema replication
 */

import { fetchJSON } from './client';

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

export interface ExtractRequest {
  document?: string;
  documents?: string[];
  text_or_documents?: string | string[];
  prompt_description?: string;
  model_id?: string;
  examples?: ExampleData[];
  api_key?: string;
}

export interface ExampleData {
  text: string;
  extractions: ExampleExtraction[];
}

export interface ExampleExtraction {
  extraction_class: string;
  extraction_text: string;
  attributes?: Record<string, any>;
}

export interface ExtractionResult {
  extraction_class: string;
  extraction_text: string;
  attributes?: Record<string, any>;
  start_index?: number;
  end_index?: number;
}

export interface ExtractResponse {
  entities: Record<string, string[]>;
  extractions: ExtractionResult[];
}

export interface KnowledgeGraphRequest {
  project_id: string;
  system_id?: string;
  sql_queries?: string[];
  json_tables?: string[];
  hive_ddls?: string[];
  control_m_files?: string[];
  options?: Record<string, any>;
}

export interface KnowledgeGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata?: {
    project_id: string;
    system_id?: string;
    total_nodes: number;
    total_edges: number;
  };
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  properties?: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  properties?: Record<string, any>;
}

export interface OCRExtractRequest {
  image_url?: string;
  image_base64?: string;
  language?: string;
  options?: Record<string, any>;
}

export interface OCRExtractResponse {
  text: string;
  confidence?: number;
  metadata?: Record<string, any>;
}

export interface SchemaReplicationRequest {
  connection_string: string;
  database?: string;
  schema?: string;
  tables?: string[];
  include_views?: boolean;
  options?: Record<string, any>;
}

export interface SchemaReplicationResponse {
  schema: {
    database: string;
    schema: string;
    tables: Array<{
      name: string;
      columns: Array<{
        name: string;
        type: string;
        nullable?: boolean;
        default?: any;
      }>;
      primary_keys?: string[];
      foreign_keys?: Array<{
        column: string;
        referenced_table: string;
        referenced_column: string;
      }>;
    }>;
    views?: Array<{
      name: string;
      definition?: string;
      columns: Array<{
        name: string;
        type: string;
      }>;
    }>;
  };
  metadata?: Record<string, any>;
}

export interface ExtractJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  source_type: string;
  created_at: string;
  completed_at?: string;
  error?: string;
  result?: ExtractResponse | KnowledgeGraphResponse;
  metadata?: Record<string, any>;
}

/**
 * Extract entities from text or documents
 */
export async function extractEntities(request: ExtractRequest): Promise<ExtractResponse> {
  return fetchJSON<ExtractResponse>('/extract', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Process knowledge graph from various sources
 */
export async function processKnowledgeGraph(
  request: KnowledgeGraphRequest
): Promise<KnowledgeGraphResponse> {
  return fetchJSON<KnowledgeGraphResponse>('/knowledge-graph', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Extract text from images using OCR
 */
export async function extractOCR(request: OCRExtractRequest): Promise<OCRExtractResponse> {
  return fetchJSON<OCRExtractResponse>('/extract/ocr', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Replicate schema from a database
 */
export async function replicateSchema(
  request: SchemaReplicationRequest
): Promise<SchemaReplicationResponse> {
  return fetchJSON<SchemaReplicationResponse>('/extract/schema-replication', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get extraction job status (if job tracking is implemented)
 */
export async function getExtractJob(jobId: string): Promise<ExtractJob> {
  return fetchJSON<ExtractJob>(`/extract/jobs/${jobId}`);
}

/**
 * List extraction jobs
 */
export async function listExtractJobs(params?: {
  limit?: number;
  offset?: number;
  status?: string;
}): Promise<ExtractJob[]> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.append('limit', params.limit.toString());
  if (params?.offset) queryParams.append('offset', params.offset.toString());
  if (params?.status) queryParams.append('status', params.status);
  
  const query = queryParams.toString();
  return fetchJSON<ExtractJob[]>(`/extract/jobs${query ? `?${query}` : ''}`);
}

