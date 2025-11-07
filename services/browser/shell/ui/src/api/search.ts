import { API_BASE } from "./client";

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
}

export interface UnifiedSearchResult extends SearchResult {
  source: string;
  score: number;
  metadata?: Record<string, unknown>;
  citations?: string[];
}

export interface UnifiedSearchResponse {
  query: string;
  sources: Record<string, unknown>;
  combined_results: UnifiedSearchResult[];
  total_count: number;
}

export async function searchDocuments(request: SearchRequest): Promise<SearchResponse> {
  const url = `${API_BASE}/search/v1/search`;
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      body: JSON.stringify({
        query: request.query,
        top_k: request.top_k ?? 10
      })
    });

    if (!response.ok) {
      let errorMessage = `Search request failed (${response.status})`;
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
        return { results: [] };
      }
      return JSON.parse(text) as SearchResponse;
    }

    return (await response.json()) as SearchResponse;
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
        use_perplexity: request.use_perplexity ?? false
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
        return { query: request.query, sources: {}, combined_results: [], total_count: 0 };
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

