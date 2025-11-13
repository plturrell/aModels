import { useState, useCallback, useRef } from 'react';
import { aiService, AIRequest, GNNRequest, GooseRequest, DeepResearchRequest } from '../services/AIIntegration';

export interface UseAIOptions {
  cache?: boolean;
  retry?: number;
  timeout?: number;
}

export interface UseAIState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  execute: (request: AIRequest) => Promise<T>;
  cancel: () => void;
}

export function useAI<T = any>(options: UseAIOptions = {}): UseAIState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(async (request: AIRequest): Promise<T> => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setLoading(true);
    setError(null);

    try {
      const result = await aiService.query(request, abortControllerRef.current.signal);
      setData(result.data as T);
      return result.data as T;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [options.cache, options.retry, options.timeout]);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
    }
  }, []);

  return { data, loading, error, execute, cancel };
}

export function useGNNAnalysis() {
  const { data, loading, error, execute } = useAI();

  const analyzeGraph = useCallback(async (request: GNNRequest) => {
    return execute({
      type: 'gnn',
      query: JSON.stringify(request),
      context: { task: request.task, payload: request }
    });
  }, [execute]);

  return { data, loading, error, analyzeGraph };
}

export function useGooseTask() {
  const { data, loading, error, execute } = useAI();

  const executeTask = useCallback(async (request: GooseRequest) => {
    return execute({
      type: 'goose',
      query: request.task,
      context: request.context,
      parameters: request.autoRemediate != null ? { autoRemediate: request.autoRemediate } : undefined
    });
  }, [execute]);

  return { data, loading, error, executeTask };
}

export function useDeepResearch() {
  const { data, loading, error, execute } = useAI();

  const research = useCallback(async (request: DeepResearchRequest) => {
    return execute({
      type: 'deep-research',
      query: request.query,
      context: { scope: request.scope, sources: request.sources }
    });
  }, [execute]);

  return { data, loading, error, research };
}

export function useHybridAI() {
  const { data, loading, error, execute } = useAI();

  const query = useCallback(async (query: string, context: Record<string, any>) => {
    return execute({
      type: 'hybrid',
      query,
      context
    });
  }, [execute]);

  return { data, loading, error, query };
}
