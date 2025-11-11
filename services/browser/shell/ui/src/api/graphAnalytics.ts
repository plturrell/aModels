/**
 * Phase 3.2: Graph Analytics API Client
 * 
 * Client for graph analytics endpoints
 */

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

export interface CommunityDetectionResponse {
  algorithm: string;
  num_communities: number;
  communities: Array<{
    size: number;
    nodes?: string[];
  }>;
  total_nodes: number;
}

export interface CentralityResponse {
  metric_type: string;
  top_k: number;
  nodes: Array<{
    node_id: string;
    label?: string;
    centrality: number;
  }>;
}

export interface GrowthTrendsResponse {
  days: number;
  trends: Array<{
    date: string;
    count: number;
  }>;
  total: number;
}

/**
 * Detect communities in the graph
 */
export async function detectCommunities(
  projectId: string,
  systemId?: string,
  algorithm: 'louvain' | 'leiden' | 'label_propagation' = 'louvain'
): Promise<CommunityDetectionResponse> {
  const url = `${API_BASE}/graph/analytics/communities?project_id=${encodeURIComponent(projectId)}${systemId ? `&system_id=${encodeURIComponent(systemId)}` : ''}&algorithm=${algorithm}`;
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Community detection request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Community detection error:', error);
    throw error;
  }
}

/**
 * Get centrality metrics
 */
export async function getCentrality(
  projectId: string,
  systemId?: string,
  type: 'degree' | 'betweenness' | 'closeness' | 'pagerank' = 'degree',
  topK: number = 20
): Promise<CentralityResponse> {
  const url = `${API_BASE}/graph/analytics/centrality?project_id=${encodeURIComponent(projectId)}${systemId ? `&system_id=${encodeURIComponent(systemId)}` : ''}&type=${type}&top_k=${topK}`;
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Centrality request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Centrality error:', error);
    throw error;
  }
}

/**
 * Get growth trends
 */
export async function getGrowthTrends(
  projectId: string,
  systemId?: string,
  days: number = 30
): Promise<GrowthTrendsResponse> {
  const url = `${API_BASE}/graph/analytics/growth?project_id=${encodeURIComponent(projectId)}${systemId ? `&system_id=${encodeURIComponent(systemId)}` : ''}&days=${days}`;
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Growth trends request failed (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Growth trends error:', error);
    throw error;
  }
}

