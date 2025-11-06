import { useEffect, useMemo, useState } from "react";

export const API_BASE = import.meta.env.VITE_SHELL_API ?? "";

async function fetchJSON<T>(endpoint: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: { Accept: "application/json" },
    ...init
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Request failed (${response.status}): ${message}`);
  }

  return (await response.json()) as T;
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
