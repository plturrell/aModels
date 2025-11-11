/**
 * Data loader for knowledge graph queries
 * Fetches knowledge graph data from Perplexity API
 */

const API_BASE = process.env.PERPLEXITY_API_BASE || "http://localhost:8000";

const QUALITY_THRESHOLDS = {
  excellent: 0.9,
  good: 0.75,
  fair: 0.5
};

function deriveQuality(score) {
  if (score == null || Number.isNaN(score)) {
    return undefined;
  }

  if (score >= QUALITY_THRESHOLDS.excellent) {
    return { score, level: "excellent" };
  }
  if (score >= QUALITY_THRESHOLDS.good) {
    return { score, level: "good" };
  }
  if (score >= QUALITY_THRESHOLDS.fair) {
    return { score, level: "fair" };
  }
  return { score, level: "poor" };
}

function buildGraphDataFromRelationships(payload) {
  const relationships = Array.isArray(payload?.relationships) ? payload.relationships : [];
  const nodes = new Map();
  const edges = [];

  relationships.forEach((rel, index) => {
    const sourceId = rel.source_id || rel.source || rel.sourceId;
    const targetId = rel.target_id || rel.target || rel.targetId;
    if (!sourceId || !targetId) {
      return;
    }

    if (!nodes.has(sourceId)) {
      nodes.set(sourceId, {
        id: sourceId,
        type: rel.source_type || rel.sourceType || "entity",
        label: rel.source_title || rel.sourceLabel || sourceId,
        properties: {
          title: rel.source_title || rel.sourceLabel,
          type: rel.source_type || rel.sourceType
        }
      });
    }

    if (!nodes.has(targetId)) {
      nodes.set(targetId, {
        id: targetId,
        type: rel.target_type || rel.targetType || "entity",
        label: rel.target_title || rel.targetLabel || targetId,
        properties: {
          title: rel.target_title || rel.targetLabel,
          type: rel.target_type || rel.targetType
        }
      });
    }

    edges.push({
      id: rel.id || `edge-${index}`,
      source: sourceId,
      target: targetId,
      label: rel.relationship_type || rel.type || "related",
      properties: {
        confidence: typeof rel.confidence === "number" ? rel.confidence : undefined,
        description: rel.description,
        metadata: rel.metadata,
        source_title: rel.source_title,
        target_title: rel.target_title
      }
    });
  });

  const avgConfidence = relationships.length
    ? relationships
        .map((rel) => (typeof rel.confidence === "number" ? rel.confidence : null))
        .filter((value) => value != null)
        .reduce((sum, value, _, arr) => sum + value / arr.length, 0)
    : undefined;

  const metadata = {
    request_id: payload?.request_id,
    query: payload?.query,
    relationship_count: relationships.length
  };

  const graphData = {
    nodes: Array.from(nodes.values()),
    edges,
    metadata,
    quality: deriveQuality(avgConfidence)
  };

  return { graphData, relationships };
}

function normalizeGraphPayload(payload) {
  if (!payload) {
    return {
      graphData: { nodes: [], edges: [], metadata: {} },
      relationships: []
    };
  }

  // Different services may already return unified graph data
  if (payload.graph_data && Array.isArray(payload.graph_data.nodes)) {
    return {
      graphData: payload.graph_data,
      relationships: Array.isArray(payload.relationships) ? payload.relationships : []
    };
  }

  if (Array.isArray(payload.nodes) && Array.isArray(payload.edges)) {
    return {
      graphData: {
        nodes: payload.nodes,
        edges: payload.edges,
        metadata: payload.metadata,
        quality: payload.quality
      },
      relationships: Array.isArray(payload.relationships) ? payload.relationships : []
    };
  }

  if (payload.results?.graph_data) {
    return {
      graphData: payload.results.graph_data,
      relationships: Array.isArray(payload.results.relationships)
        ? payload.results.relationships
        : []
    };
  }

  return buildGraphDataFromRelationships(payload);
}

async function fetchGraphRelationships(requestId) {
  const url = `${API_BASE}/api/perplexity/graph/${requestId}/relationships`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch relationships: ${response.statusText}`);
  }

  return response.json();
}

async function fetchGraphQuery(requestId, query, params = {}) {
  const url = `${API_BASE}/api/perplexity/graph/${requestId}/query`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, params })
  });

  if (!response.ok) {
    throw new Error(`Graph query failed: ${response.statusText}`);
  }

  return response.json();
}

export default async function loadGraph(requestId, query = null, params = {}) {
  try {
    const payload = query
      ? await fetchGraphQuery(requestId, query, params)
      : await fetchGraphRelationships(requestId);

    const normalized = normalizeGraphPayload(payload);
    return {
      graphData: normalized.graphData,
      relationships: normalized.relationships,
      raw: payload
    };
  } catch (error) {
    console.error("Error loading graph data:", error);
    return {
      graphData: { nodes: [], edges: [], metadata: {} },
      relationships: [],
      raw: null,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

