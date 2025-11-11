/**
 * TypeScript API client for Narrative GNN endpoints
 */

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';
const TRAINING_SERVICE_URL = import.meta.env.VITE_TRAINING_SERVICE_URL || 'http://localhost:8080';

export interface NarrativeExplainRequest {
  nodes: Array<{ id: string; type?: string; label?: string; properties?: Record<string, any> }>;
  edges: Array<{ source_id: string; target_id: string; label?: string; type?: string; properties?: Record<string, any> }>;
  storyline_id?: string;
  focus_node_id?: string;
  current_time?: number;
}

export interface NarrativeExplainResponse {
  status: string;
  explanation: string;
  key_actors: Array<{ node_id: string; influence?: number }>;
  turning_points: Array<{ node_id: string; significance?: number }>;
  causal_chain: Array<{ source: string; target: string; label?: string }>;
  timestamp: string;
}

export interface NarrativePredictRequest {
  nodes: Array<{ id: string; type?: string; label?: string; properties?: Record<string, any> }>;
  edges: Array<{ source_id: string; target_id: string; label?: string; type?: string; properties?: Record<string, any> }>;
  storyline_id?: string;
  current_time: number;
  future_time?: number;
  num_trajectories?: number;
}

export interface NarrativePredictResponse {
  status: string;
  predictions: Array<any>;
  trajectories: Array<Array<any>>;
  scores: Array<number>;
  timestamp: string;
}

export interface NarrativeAnomalyRequest {
  nodes: Array<{ id: string; type?: string; label?: string; properties?: Record<string, any> }>;
  edges: Array<{ source_id: string; target_id: string; label?: string; type?: string; properties?: Record<string, any> }>;
  storyline_id?: string;
  current_time?: number;
  threshold?: number;
}

export interface NarrativeAnomalyResponse {
  status: string;
  anomalies: Array<any>;
  violations: Array<any>;
  inconsistencies: Array<any>;
  timestamp: string;
}

export interface NarrativeMCTSRequest {
  nodes: Array<{ id: string; type?: string; label?: string; properties?: Record<string, any> }>;
  edges: Array<{ source_id: string; target_id: string; label?: string; type?: string; properties?: Record<string, any> }>;
  storyline_id?: string;
  current_time: number;
  num_rollouts?: number;
  max_depth?: number;
  exploration_c?: number;
}

export interface NarrativeMCTSResponse {
  status: string;
  best_path: Array<any>;
  path_value: number;
  explored_paths: Array<Array<any>>;
  rollouts: number;
  timestamp: string;
}

export interface NarrativeStorylineRequest {
  nodes: Array<{ id: string; type?: string; label?: string; properties?: Record<string, any> }>;
  edges: Array<{ source_id: string; target_id: string; label?: string; type?: string; properties?: Record<string, any> }>;
  storyline_id?: string;
  operation: 'list' | 'get' | 'key_actors' | 'turning_points' | 'causal_chain';
}

export interface NarrativeStorylineResponse {
  status: string;
  operation: string;
  storylines?: string[] | Record<string, any>;
  storyline?: any;
  key_actors?: Array<{ node_id: string; influence: number }>;
  turning_points?: Array<{ node_id: string; significance: number }>;
  causal_chain?: Array<{ source: string; target: string; label?: string }>;
  timestamp: string;
}

/**
 * Generate narrative explanation
 */
export async function explainNarrative(request: NarrativeExplainRequest): Promise<NarrativeExplainResponse> {
  const url = `${TRAINING_SERVICE_URL}/narrative/explain`;
  
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
      throw new Error(`Narrative explanation request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Narrative explanation error:', error);
    throw error;
  }
}

/**
 * Predict future narrative states
 */
export async function predictNarrative(request: NarrativePredictRequest): Promise<NarrativePredictResponse> {
  const url = `${TRAINING_SERVICE_URL}/narrative/predict`;
  
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
      throw new Error(`Narrative prediction request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Narrative prediction error:', error);
    throw error;
  }
}

/**
 * Detect narrative anomalies
 */
export async function detectNarrativeAnomalies(request: NarrativeAnomalyRequest): Promise<NarrativeAnomalyResponse> {
  const url = `${TRAINING_SERVICE_URL}/narrative/detect-anomalies`;
  
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
      throw new Error(`Narrative anomaly detection request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Narrative anomaly detection error:', error);
    throw error;
  }
}

/**
 * Perform MCTS what-if analysis
 */
export async function narrativeMCTS(request: NarrativeMCTSRequest): Promise<NarrativeMCTSResponse> {
  const url = `${TRAINING_SERVICE_URL}/narrative/mcts`;
  
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
      throw new Error(`MCTS analysis request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('MCTS analysis error:', error);
    throw error;
  }
}

/**
 * Perform storyline operations
 */
export async function narrativeStoryline(request: NarrativeStorylineRequest): Promise<NarrativeStorylineResponse> {
  const url = `${TRAINING_SERVICE_URL}/narrative/storyline`;
  
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
      throw new Error(`Storyline operation request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Storyline operation error:', error);
    throw error;
  }
}

