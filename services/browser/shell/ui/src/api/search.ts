import { API_BASE } from "./client";
const SEARCH_BASE = import.meta.env.VITE_SEARCH_BASE || "";

export interface SearchRequest {
  query: string;
  top_k?: number;
}

export interface SearchResult {
  id: string;
  content: string;
  similarity: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface UnifiedSearchRequest extends SearchRequest {
  sources?: string[];
  use_perplexity?: boolean;
  enable_framework?: boolean;  // Query understanding and result enrichment
  enable_plot?: boolean;  // Visualization data
  enable_stdlib?: boolean;  // Result processing (default: true)
  stdlib_operations?: string[];  // Operations: deduplicate, sort_by_score, truncate_content
  enable_dashboard?: boolean;
  enable_narrative?: boolean;
}

export interface UnifiedSearchResult extends SearchResult {
  source: string;
  score: number;
  metadata?: Record<string, unknown>;
  citations?: string[];
}

export interface QueryEnrichment {
  original_query: string;
  enriched_query: string;
  intent_summary?: string;
  entities?: string[];
  enriched: boolean;
}

export interface ResultEnrichment {
  summary?: string;
  insights?: string[];
  enriched: boolean;
}

export interface VisualizationData {
  source_distribution: Record<string, number>;
  score_statistics: {
    average: number;
    min: number;
    max: number;
    count: number;
  };
  timeline?: Array<{
    timestamp: string;
    score: number;
    source: string;
  }>;
  total_results: number;
}

export interface DashboardChart {
  type: string;  // bar, line, pie, scatter, heatmap, network
  title: string;
  data_source: string;
  x_axis?: string;
  y_axis?: string;
  config?: Record<string, unknown>;
}

export interface DashboardMetric {
  label: string;
  value: string | number;
  format?: string;  // number, percentage, currency
}

export interface DashboardSpecification {
  title: string;
  description: string;
  charts: DashboardChart[];
  metrics: DashboardMetric[];
  insights: string[];
}

export interface Dashboard {
  specification: DashboardSpecification;
  enriched: boolean;
  error?: string;
}

export interface Narrative {
  markdown: string;
  sections: Record<string, string>;
  enriched: boolean;
}

export interface UnifiedSearchResponse {
  query: string;
  sources: Record<string, unknown>;
  combined_results: UnifiedSearchResult[];
  total_count: number;
  query_enrichment?: QueryEnrichment;
  result_enrichment?: ResultEnrichment;
  visualization?: VisualizationData;
  dashboard?: Dashboard;
  narrative?: Narrative;
  metadata: {
    sources_queried: number;
    sources_successful: number;
    sources_failed: number;
    execution_time_ms: number;
  };
}

export async function searchDocuments(request: SearchRequest): Promise<SearchResponse> {
  const direct = SEARCH_BASE ? `${SEARCH_BASE}/v1/search` : "";
  const url = direct || `${API_BASE}/search/unified`;
  try {
    // Attempt direct search first if configured
    if (direct) {
      try {
        const directResp = await fetch(direct, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({ query: request.query, top_k: request.top_k ?? 10 })
        });
        if (directResp.ok) {
          const text = await directResp.text();
          if (!text.trim()) return { results: [] };
          return JSON.parse(text) as SearchResponse;
        }
        // Fall through to unified on non-OK
      } catch (e) {
        // Network failure -> fall back to unified
      }
    }

    // Unified search via gateway (uses defaults for available sources)
    const unifiedResp = await fetch(`${API_BASE}/search/unified`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        query: request.query,
        top_k: request.top_k ?? 10,
        // Do not force sources to allow Gateway defaults (inference, knowledge_graph, catalog)
        use_perplexity: false,
        enable_framework: false,
        enable_plot: false,
        enable_stdlib: false,
      })
    });
    if (!unifiedResp.ok) {
      let errorMessage = `Search request failed (${unifiedResp.status})`;
      try {
        const errorText = await unifiedResp.text();
        if (errorText) {
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorText;
          } catch {
            errorMessage = errorText;
          }
        }
      } catch {
        errorMessage = `HTTP ${unifiedResp.status} ${unifiedResp.statusText}`;
      }
      throw new Error(errorMessage);
    }
    const contentType = unifiedResp.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const text = await unifiedResp.text();
      if (!text.trim()) {
        return { results: [] };
      }
      const data = JSON.parse(text);
      const combined = Array.isArray(data?.combined_results) ? data.combined_results : [];
      const results = combined.map((r: any) => ({
        id: r.id ?? "",
        content: r.content ?? "",
        similarity: typeof r.score === "number" ? r.score : r.similarity ?? 0
      }));
      return { results };
    }
    const data = await unifiedResp.json();
    const combined = Array.isArray(data?.combined_results) ? data.combined_results : [];
    const results = combined.map((r: any) => ({
      id: r.id ?? "",
      content: r.content ?? "",
      similarity: typeof r.score === "number" ? r.score : r.similarity ?? 0
    }));
    return { results };
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach search service at ${url}. Check if the service is running and accessible.`);
      }
      throw error;
    }
    throw new Error(`Unexpected error: ${String(error)}`);
  }
}

// Stub functions for narrative and dashboard generation
export async function generateNarrative(requestIdOrQuery: string, searchResponse?: UnifiedSearchResponse): Promise<{ narrative: Narrative }> {
  throw new Error("generateNarrative not implemented");
}

export async function generateDashboard(requestIdOrQuery: string, searchResponse?: UnifiedSearchResponse): Promise<{ dashboard: Dashboard }> {
  throw new Error("generateDashboard not implemented");
}

export async function generateNarrativeAndDashboard(requestIdOrQuery: string, searchResponse?: UnifiedSearchResponse): Promise<{ narrative: Narrative; dashboard: Dashboard; search_metadata?: any }> {
  throw new Error("generateNarrativeAndDashboard not implemented");
}

export async function exportNarrativeToPowerPoint(requestIdOrQuery: string, narrative?: string, metadata?: any): Promise<Blob> {
  throw new Error("exportNarrativeToPowerPoint not implemented");
}

export async function exportDashboardToPowerPoint(requestIdOrQuery: string, dashboard?: any, metadata?: any): Promise<Blob> {
  throw new Error("exportDashboardToPowerPoint not implemented");
}

export async function exportNarrativeAndDashboardToPowerPoint(requestIdOrQuery: string, narrative?: string, dashboard?: any, metadata?: any): Promise<Blob> {
  throw new Error("exportNarrativeAndDashboardToPowerPoint not implemented");
}

export async function unifiedSearch(request: UnifiedSearchRequest): Promise<UnifiedSearchResponse> {
  const url = `${API_BASE}/search/unified`;
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      body: JSON.stringify({
        query: request.query,
        top_k: request.top_k ?? 20,
        sources: request.sources ?? ["inference", "knowledge_graph", "catalog"],
        use_perplexity: request.use_perplexity ?? false,
        enable_framework: request.enable_framework ?? false,
        enable_plot: request.enable_plot ?? false,
        enable_stdlib: request.enable_stdlib ?? true,
        stdlib_operations: request.stdlib_operations ?? ["deduplicate", "sort_by_score"],
        enable_dashboard: request.enable_dashboard ?? false,
        enable_narrative: request.enable_narrative ?? false
      })
    });

    if (!response.ok) {
      let errorMessage = `Unified search request failed (${response.status})`;
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
        return { query: request.query, sources: {}, combined_results: [], total_count: 0, metadata: { sources_queried: 0, sources_successful: 0, sources_failed: 0, execution_time_ms: 0 } };
      }
      return JSON.parse(text) as UnifiedSearchResponse;
    }

    return (await response.json()) as UnifiedSearchResponse;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach unified search at ${url}. Check if the gateway service is running and accessible.`);
      }
      throw error;
    }
    throw new Error(`Unexpected error: ${String(error)}`);
  }
}

