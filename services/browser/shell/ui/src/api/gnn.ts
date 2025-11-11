/**
 * Phase 3: GNN API Client
 * 
 * Client for interacting with training service GNN endpoints
 */

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';
const TRAINING_SERVICE_URL = import.meta.env.VITE_TRAINING_SERVICE_URL || 'http://localhost:8080';

export interface GNNEmbeddingsRequest {
  nodes: Array<{
    id: string;
    type?: string;
    label?: string;
    properties?: Record<string, any>;
  }>;
  edges: Array<{
    source_id: string;
    target_id: string;
    label?: string;
    type?: string;
    properties?: Record<string, any>;
  }>;
  graph_level?: boolean;
}

export interface GNNEmbeddingsResponse {
  status: string;
  graph_embedding?: number[];
  node_embeddings?: Record<string, number[]>;
  embedding_dim?: number;
  cached?: boolean;
  timestamp?: string;
}

export interface GNNClassifyRequest {
  nodes: Array<{
    id: string;
    type?: string;
    label?: string;
    properties?: Record<string, any>;
  }>;
  edges: Array<{
    source_id: string;
    target_id: string;
    label?: string;
    type?: string;
    properties?: Record<string, any>;
  }>;
}

export interface GNNClassifyResponse {
  status: string;
  classifications?: Array<{
    node_id: string;
    predicted_class: string;
    confidence: number;
    probabilities?: Record<string, number>;
  }>;
  num_classes?: number;
  cached?: boolean;
  timestamp?: string;
}

export interface GNNPredictLinksRequest {
  nodes: Array<{
    id: string;
    type?: string;
    label?: string;
    properties?: Record<string, any>;
  }>;
  edges: Array<{
    source_id: string;
    target_id: string;
    label?: string;
    type?: string;
    properties?: Record<string, any>;
  }>;
  top_k?: number;
}

export interface GNNPredictLinksResponse {
  status: string;
  predictions?: Array<{
    source_id: string;
    target_id: string;
    probability: number;
    predicted_label?: string;
  }>;
  top_k?: number;
  cached?: boolean;
  timestamp?: string;
}

export interface GNNStructuralInsightsRequest {
  nodes: Array<{
    id: string;
    type?: string;
    label?: string;
    properties?: Record<string, any>;
  }>;
  edges: Array<{
    source_id: string;
    target_id: string;
    label?: string;
    type?: string;
    properties?: Record<string, any>;
  }>;
  insight_type?: 'anomalies' | 'patterns' | 'all';
  threshold?: number;
}

export interface GNNStructuralInsightsResponse {
  status: string;
  insights?: {
    anomalies?: {
      anomalous_nodes?: Array<{
        node_id: string;
        anomaly_score: number;
        reason?: string;
      }>;
      anomalous_edges?: Array<{
        source_id: string;
        target_id: string;
        anomaly_score: number;
        reason?: string;
      }>;
      num_anomalies?: number;
    };
    patterns?: {
      graph_embedding_dim?: number;
      num_node_embeddings?: number;
      embedding_available?: boolean;
    };
    node_types?: {
      num_classified?: number;
      num_classes?: number;
    };
  };
  insight_type?: string;
  cached?: boolean;
  timestamp?: string;
}

/**
 * Get GNN embeddings for nodes and edges
 */
export async function getGNNEmbeddings(request: GNNEmbeddingsRequest): Promise<GNNEmbeddingsResponse> {
  const url = `${TRAINING_SERVICE_URL}/gnn/embeddings`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GNN embeddings request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('GNN embeddings error:', error);
    throw error;
  }
}

/**
 * Classify nodes using GNN
 */
export async function classifyNodes(request: GNNClassifyRequest): Promise<GNNClassifyResponse> {
  const url = `${TRAINING_SERVICE_URL}/gnn/classify`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GNN classification request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('GNN classification error:', error);
    throw error;
  }
}

/**
 * Predict links using GNN
 */
export async function predictLinks(request: GNNPredictLinksRequest): Promise<GNNPredictLinksResponse> {
  const url = `${TRAINING_SERVICE_URL}/gnn/predict-links`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GNN link prediction request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('GNN link prediction error:', error);
    throw error;
  }
}

/**
 * Get structural insights (anomalies, patterns) using GNN
 */
export async function getStructuralInsights(request: GNNStructuralInsightsRequest): Promise<GNNStructuralInsightsResponse> {
  const url = `${TRAINING_SERVICE_URL}/gnn/structural-insights`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GNN structural insights request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('GNN structural insights error:', error);
    throw error;
  }
}

