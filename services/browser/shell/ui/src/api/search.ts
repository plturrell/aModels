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

