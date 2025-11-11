/**
 * Graph data types for visualization and API
 */

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
