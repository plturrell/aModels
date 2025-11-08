import { useEffect, useMemo, useState } from "react";

export const API_BASE = import.meta.env.VITE_GATEWAY_BASE || "http://localhost:8000";
export const PERPLEXITY_API_BASE = import.meta.env.VITE_PERPLEXITY_API_BASE || "http://localhost:8000";

async function fetchJSON<T>(endpoint: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  
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
      let errorMessage = `Request failed (${response.status})`;
      try {
        const errorText = await response.text();
        if (errorText) {
          // Try to parse as JSON for structured error messages
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

    // Handle empty responses
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
      // Enhance network errors with more context
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach ${url}. Check if the service is running and accessible.`);
      }
      throw error;
    }
    throw new Error(`Unexpected error: ${String(error)}`);
  }
}

export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refresh: () => void;
}

export function useApiData<T, R = T>(
  endpoint: string,
  transform?: (value: T) => R,
  deps: unknown[] = []
): ApiState<R> {
  const [data, setData] = useState<R | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const refresh = useMemo(() => () => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    setLoading(true);
    setError(null);

    fetchJSON<T>(endpoint, { signal: controller.signal })
      .then((value) => {
        if (!isMounted) return;
        const next = transform ? transform(value) : ((value as unknown) as R);
        setData(next);
      })
      .catch((err) => {
        if (!isMounted || controller.signal.aborted) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        setData(null);
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [endpoint, reloadToken, transform, ...deps]);

  return { data, loading, error, refresh };
}

export { fetchJSON };
