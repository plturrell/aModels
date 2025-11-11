/**
 * Phase 1.2: Graph API Client
 * 
 * Client for interacting with graph service endpoints for visualization and exploration
 */

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';
const GRAPH_SERVICE_URL = import.meta.env.VITE_GRAPH_SERVICE_URL || 'http://localhost:8081';

export interface GraphVisualizeRequest {
  project_id: string;
  system_id?: string;
  node_types?: string[];
  edge_types?: string[];
  limit?: number;
  depth?: number;
}

export interface GraphVisualizeResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: {
    total_nodes: number;
    total_edges: number;
    node_types: Record<string, number>;
    edge_types: Record<string, number>;
  };
}

export interface GraphExploreRequest {
  node_id: string;
  depth?: number;
  direction?: 'outgoing' | 'incoming' | 'both';
  limit?: number;
}

export interface GraphExploreResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  paths: Array<{
    nodes: string[];
    edges: string[];
    length: number;
  }>;
}

export interface GraphStatsResponse {
  total_nodes: number;
  total_edges: number;
  node_types: Record<string, number>;
  edge_types: Record<string, number>;
  density: number;
  average_degree: number;
  largest_component_size: number;
  communities: number;
}

export interface GraphQueryRequest {
  query: string; // Cypher query
  params?: Record<string, any>;
}

export interface GraphQueryResponse {
  columns: string[];
  data: Array<Record<string, any>>;
  execution_time_ms: number;
}

export interface GraphPathRequest {
  source_id: string;
  target_id: string;
  max_depth?: number;
  relationship_types?: string[];
}

export interface GraphPathResponse {
  paths: Array<{
    nodes: string[];
    edges: string[];
    length: number;
    weight?: number;
  }>;
  shortest_path?: {
    nodes: string[];
    edges: string[];
    length: number;
  };
}

export interface GraphNode {
  id: string;
  type?: string;
  label?: string;
  properties?: Record<string, any>;
}

export interface GraphEdge {
  id?: string;
  source_id: string;
  target_id: string;
  label?: string;
  type?: string;
  properties?: Record<string, any>;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/**
 * Get graph data for visualization
 */
export async function visualizeGraph(request: GraphVisualizeRequest): Promise<GraphVisualizeResponse> {
  const url = `${API_BASE}/graph/visualize`;
  
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
      throw new Error(`Graph visualization request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Graph visualization error:', error);
    throw error;
  }
}

/**
 * Explore graph from a specific node
 */
export async function exploreGraph(request: GraphExploreRequest): Promise<GraphExploreResponse> {
  const url = `${API_BASE}/graph/explore`;
  
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
      throw new Error(`Graph exploration request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Graph exploration error:', error);
    throw error;
  }
}

/**
 * Get graph statistics
 */
export async function getGraphStats(project_id: string, system_id?: string): Promise<GraphStatsResponse> {
  const url = `${API_BASE}/graph/stats?project_id=${encodeURIComponent(project_id)}${system_id ? `&system_id=${encodeURIComponent(system_id)}` : ''}`;
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Graph stats request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Graph stats error:', error);
    throw error;
  }
}

/**
 * Execute a Cypher query
 */
export async function queryGraph(request: GraphQueryRequest): Promise<GraphQueryResponse> {
  const url = `${API_BASE}/graph/query`;
  
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
      throw new Error(`Graph query request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Graph query error:', error);
    throw error;
  }
}

/**
 * Find paths between two nodes
 */
export async function findPaths(request: GraphPathRequest): Promise<GraphPathResponse> {
  const url = `${API_BASE}/graph/paths`;
  
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
      throw new Error(`Graph path request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Graph path error:', error);
    throw error;
  }
}

