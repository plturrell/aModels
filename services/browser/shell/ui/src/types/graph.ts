export interface GraphNodeProperties {
  title?: string;
  type?: string;
  group?: number;
  [key: string]: unknown;
}

export interface GraphNode {
  id: string;
  type?: string;
  label?: string;
  properties?: GraphNodeProperties;
}

export interface GraphEdgeProperties {
  confidence?: number;
  description?: string;
  [key: string]: unknown;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  properties?: GraphEdgeProperties;
}

export interface GraphQuality {
  score: number;
  level: string;
  issues?: string[];
  recommendations?: string[];
  processing_strategy?: string;
}

export interface GraphMetadata {
  request_id?: string;
  query?: string;
  relationship_count?: number;
  [key: string]: unknown;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata?: GraphMetadata;
  quality?: GraphQuality;
}

export interface GraphPayload {
  graphData: GraphData;
  relationships?: unknown[];
  raw?: unknown;
  error?: string;
}

export function isGraphNode(value: unknown): value is GraphNode {
  if (!value || typeof value !== "object") return false;
  const node = value as GraphNode;
  return typeof node.id === "string";
}

export function isGraphEdge(value: unknown): value is GraphEdge {
  if (!value || typeof value !== "object") return false;
  const edge = value as GraphEdge;
  return typeof edge.source === "string" && typeof edge.target === "string";
}

export function isGraphData(value: unknown): value is GraphData {
  if (!value || typeof value !== "object") return false;
  const data = value as GraphData;
  return Array.isArray(data.nodes) && Array.isArray(data.edges);
}

export function ensureGraphData(value: unknown): GraphData {
  if (isGraphData(value)) {
    return {
      nodes: value.nodes.map((node) => ({
        id: node.id,
        type: node.type,
        label: node.label,
        properties: node.properties
      })),
      edges: value.edges
        .map((edge) => {
          if (!edge.source || !edge.target) return null;
          return {
            id: edge.id || `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            label: edge.label,
            properties: edge.properties
          };
        })
        .filter((edge): edge is GraphEdge => Boolean(edge)),
      metadata: value.metadata,
      quality: value.quality
    };
  }

  return { nodes: [], edges: [] };
}
