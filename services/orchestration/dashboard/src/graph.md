---
title: Knowledge Graph
description: Interactive knowledge graph visualization
---

# Knowledge Graph Dashboard

<div class="dashboard-header">
  <h1>Knowledge Graph</h1>
  <p>Explore relationships and connections discovered in processed documents</p>
</div>

```js
import {loadGraph} from "../data/loaders/graph.js";
import {loadIntelligence} from "../data/loaders/intelligence.js";
```

```js
// Get request ID from URL parameters (deep linking support)
const urlParams = typeof window !== "undefined" 
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();
const urlRequestId = urlParams.get("request_id") || "";

// Input for request ID (if not in URL)
const requestId = typeof Inputs !== "undefined" && !urlRequestId
  ? await Inputs.text({label: "Request ID", value: urlRequestId, placeholder: "Enter request ID or use ?request_id=xxx in URL"})
  : urlRequestId || "demo";
```

```js
// Load graph data
const { graphData, relationships: legacyRelationships } = await loadGraph(requestId);
const intelligence = await loadIntelligence(requestId);
```

```js
// Normalise nodes and edges from unified GraphData format
const rawNodes = Array.isArray(graphData?.nodes) ? graphData.nodes : [];
const rawEdges = Array.isArray(graphData?.edges) ? graphData.edges : [];

const graphNodes = rawNodes.map((node, index) => {
  const id = node.id || `node-${index}`;
  const label = node.label || node.properties?.title || id;
  const type = node.type || node.properties?.type || "entity";
  const group = typeof node.properties?.group === "number" ? node.properties.group : 1;
  return {
    id,
    label,
    type,
    group,
    raw: node
  };
});

const nodeMap = new Map(graphNodes.map(node => [node.id, node]));

const graphEdges = rawEdges
  .map((edge, index) => {
    const source = edge.source || edge.source_id;
    const target = edge.target || edge.target_id;
    if (!source || !target) {
      return null;
    }
    const confidence = typeof edge.confidence === "number"
      ? edge.confidence
      : typeof edge.properties?.confidence === "number"
        ? edge.properties.confidence
        : undefined;
    return {
      id: edge.id || `edge-${index}`,
      source,
      target,
      type: edge.label || edge.type || "related",
      strength: confidence ?? 0.5,
      value: confidence ?? 0.5,
      raw: edge
    };
  })
  .filter(Boolean);

// Legacy relationship fallback (for older payloads)
const legacy = Array.isArray(legacyRelationships) ? legacyRelationships : [];

const relationshipEntries = legacy.length > 0
  ? legacy.map(rel => ({
      source: rel.source_id || rel.source || rel.sourceId,
      target: rel.target_id || rel.target || rel.targetId,
      sourceTitle: rel.source_title || rel.sourceLabel || rel.source_id,
      targetTitle: rel.target_title || rel.targetLabel || rel.target_id,
      type: rel.relationship_type || rel.type || "related",
      confidence: typeof rel.confidence === "number" ? rel.confidence : undefined
    }))
  : graphEdges.map(edge => ({
      source: edge.source,
      target: edge.target,
      sourceTitle: nodeMap.get(edge.source)?.label || edge.source,
      targetTitle: nodeMap.get(edge.target)?.label || edge.target,
      type: edge.type,
      confidence: edge.strength
    }));
```

```js
// Graph summary statistics
const avgConfidence = relationshipEntries.length > 0
  ? relationshipEntries
      .map(rel => (typeof rel.confidence === "number" ? rel.confidence : null))
      .filter(conf => conf !== null)
      .reduce((sum, conf, _, arr) => sum + conf / arr.length, 0)
  : 0;

const graphStats = {
  nodes: graphNodes.length,
  edges: graphEdges.length,
  relationships: relationshipEntries.length,
  avgConfidence
};

const quality = graphData?.quality;
```

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">{graphStats.nodes}</div>
    <div class="stat-label">Nodes</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{graphStats.edges}</div>
    <div class="stat-label">Edges</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{graphStats.relationships}</div>
    <div class="stat-label">Relationships</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{(graphStats.avgConfidence * 100).toFixed(1)}%</div>
    <div class="stat-label">Avg Confidence</div>
  </div>
  {quality ? (
    <div class="stat-card">
      <div class="stat-value">{quality.level?.toUpperCase?.() || "N/A"}</div>
      <div class="stat-label">Quality ({(quality.score * 100).toFixed(1)}%)</div>
    </div>
  ) : null}
</div>

```js
import * as Plot from "@observablehq/plot";
```

```js
// Node type distribution
const nodeTypeData = graphNodes.reduce((acc, node) => {
  acc[node.type] = (acc[node.type] || 0) + 1;
  return acc;
}, {});

const nodeTypePlotData = Object.entries(nodeTypeData).map(([type, count]) => ({
  type,
  count
}));
```

<div class="chart-container">
  <h2>Node Distribution</h2>
  {Plot.plot({
    margin: {top: 20, right: 20, bottom: 60, left: 60},
    width: 800,
    height: 400,
    marks: [
      Plot.barY(nodeTypePlotData, {
        x: "type",
        y: "count",
        fill: "#007AFF",
        fillOpacity: 0.8
      })
    ],
    x: {
      label: "Node Type",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    y: {
      label: "Count",
      labelFontSize: 14,
      labelFontWeight: "500"
    }
  })}
</div>

```js
// Relationship type distribution
const relTypeData = graphEdges.reduce((acc, edge) => {
  acc[edge.type] = (acc[edge.type] || 0) + 1;
  return acc;
}, {});

const relTypePlotData = Object.entries(relTypeData).map(([type, count]) => ({
  type,
  count
}));
```

<div class="chart-container">
  <h2>Relationship Types</h2>
  {Plot.plot({
    margin: {top: 20, right: 20, bottom: 60, left: 60},
    width: 800,
    height: 400,
    marks: [
      Plot.barY(relTypePlotData, {
        x: "type",
        y: "count",
        fill: "#34C759",
        fillOpacity: 0.8
      })
    ],
    x: {
      label: "Relationship Type",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    y: {
      label: "Count",
      labelFontSize: 14,
      labelFontWeight: "500"
    }
  })}
</div>

```js
// Confidence distribution
const confidenceData = relationshipEntries
  .map(r => (typeof r.confidence === "number" ? r.confidence : null))
  .filter(conf => conf !== null && conf >= 0 && conf <= 1);
```

<div class="chart-container">
  <h2>Confidence Distribution</h2>
  {Plot.plot({
    margin: {top: 20, right: 20, bottom: 60, left: 60},
    width: 800,
    height: 400,
    marks: [
      Plot.rectY(confidenceData, Plot.binX({y: "count"}, {
        x: d => d,
        fill: "#FF9500",
        fillOpacity: 0.8,
        thresholds: 20
      }))
    ],
    x: {
      label: "Confidence",
      labelFontSize: 14,
      labelFontWeight: "500",
      domain: [0, 1]
    },
    y: {
      label: "Count",
      labelFontSize: 14,
      labelFontWeight: "500"
    }
  })}
</div>

<div class="relationships-list">
  <h2>Relationships</h2>
  <div class="relationships-grid">
    {relationshipEntries.slice(0, 20).map((rel, i) => html`
      <div class="relationship-card">
        <div class="relationship-source">
          <strong>${rel.sourceTitle || rel.source || "Unknown"}</strong>
          <span class="relationship-type">${rel.type || "related"}</span>
        </div>
        <div class="relationship-arrow">→</div>
        <div class="relationship-target">
          <strong>${rel.targetTitle || rel.target || "Unknown"}</strong>
        </div>
        <div class="relationship-confidence">
          ${(((typeof rel.confidence === "number" ? rel.confidence : 0.5)) * 100).toFixed(1)}% confidence
        </div>
      </div>
    `)}
  </div>
</div>

<style>
.dashboard-header {
  margin-bottom: 32px;
}

.dashboard-header h1 {
  font-size: 32px;
  font-weight: 600;
  color: #1d1d1f;
  margin: 0 0 8px 0;
}

.dashboard-header p {
  font-size: 16px;
  color: #86868b;
  margin: 0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 32px 0;
}

.stat-card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  text-align: center;
}

.stat-value {
  font-size: 32px;
  font-weight: 600;
  color: #007AFF;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #86868b;
  font-weight: 400;
}

.chart-container {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  margin: 32px 0;
}

.chart-container h2 {
  font-size: 20px;
  font-weight: 600;
  color: #1d1d1f;
  margin: 0 0 20px 0;
}

.relationships-list {
  margin: 32px 0;
}

.relationships-list h2 {
  font-size: 24px;
  font-weight: 600;
  color: #1d1d1f;
  margin: 0 0 20px 0;
}

.relationships-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.relationship-card {
  background: white;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.relationship-source,
.relationship-target {
  font-size: 14px;
}

.relationship-type {
  display: inline-block;
  background: #F2F2F7;
  color: #1d1d1f;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  margin-left: 8px;
}

.relationship-arrow {
  color: #86868b;
  font-size: 18px;
  text-align: center;
  margin: 4px 0;
}

.relationship-confidence {
  font-size: 12px;
  color: #86868b;
  text-align: right;
}
</style>

