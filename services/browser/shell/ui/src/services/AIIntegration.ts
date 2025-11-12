import { API_BASE } from "../api/client";

export interface AIRequest {
  type: 'gnn' | 'goose' | 'deep-research' | 'hybrid';
  query: string;
  context?: Record<string, any>;
  parameters?: {
    temperature?: number;
    maxTokens?: number;
    model?: string;
    [key: string]: unknown;
  };
}

export interface AIResponse<T = any> {
  data: T;
  model: string;
  confidence: number;
  citations?: Array<{
    id: string;
    title: string;
    url: string;
    snippet: string;
  }>;
  metadata: {
    latency: number;
    tokens: number;
    route: string[];
  };
}

export interface GNNRequest {
  graph: {
    nodes: Array<{
      id: string;
      properties: Record<string, any>;
    }>;
    edges: Array<{
      source: string;
      target: string;
      properties: Record<string, any>;
    }>;
  };
  task: 'embeddings' | 'classification' | 'link-prediction' | 'structural-insights';
}

export interface GooseRequest {
  task: string;
  context: Record<string, any>;
  autoRemediate?: boolean;
}

export interface DeepResearchRequest {
  query: string;
  scope: 'regulatory' | 'technical' | 'compliance';
  sources?: string[];
}

export class AIService {
  private static instance: AIService;
  private cache = new Map<string, { data: any; timestamp: number }>();
  private cacheTTL = 5 * 60 * 1000; // 5 minutes

  static getInstance(): AIService {
    if (!AIService.instance) {
      AIService.instance = new AIService();
    }
    return AIService.instance;
  }

  async query(request: AIRequest, signal?: AbortSignal): Promise<AIResponse> {
    const cacheKey = this.generateCacheKey(request);
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    const start = Date.now();
    try {
      const response = await fetch(`${API_BASE}/api/ai/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: request.type,
          query: request.query,
          context: request.context,
          parameters: request.parameters
        }),
        signal
      });

      if (!response.ok) {
        throw new Error(`AI service error: ${response.status}`);
      }

      const result = (await response.json()) as AIResponse;
      const latency = Date.now() - start;
      const processed: AIResponse = {
        ...result,
        metadata: {
          latency,
          tokens: result?.metadata?.tokens ?? 0,
          route: result?.metadata?.route ?? []
        }
      };

      this.setCache(cacheKey, processed);
      return processed;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`AI query failed: ${message}`);
    }
  }

  async gnnAnalysis(request: GNNRequest): Promise<AIResponse> {
    return this.query({
      type: 'gnn',
      query: JSON.stringify(request),
      context: { task: request.task }
    });
  }

  async gooseTask(request: GooseRequest): Promise<AIResponse> {
    return this.query({
      type: 'goose',
      query: request.task,
      context: request.context,
      parameters: { autoRemediate: request.autoRemediate }
    });
  }

  async deepResearch(request: DeepResearchRequest): Promise<AIResponse> {
    return this.query({
      type: 'deep-research',
      query: request.query,
      context: { scope: request.scope, sources: request.sources }
    });
  }

  async hybridQuery(query: string, context: Record<string, any>): Promise<AIResponse> {
    return this.query({
      type: 'hybrid',
      query,
      context
    });
  }

  private generateCacheKey(request: AIRequest): string {
    return btoa(JSON.stringify({
      type: request.type,
      query: request.query,
      context: request.context,
      parameters: request.parameters
    }));
  }

  private getFromCache(key: string): AIResponse | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: AIResponse): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  clearCache(): void {
    this.cache.clear();
  }
}

export const aiService = AIService.getInstance();
