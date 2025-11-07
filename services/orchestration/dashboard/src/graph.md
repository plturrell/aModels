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
// Get request ID from URL or default
const urlParams = new URLSearchParams(window.location.search);
const requestId = urlParams.get("request_id") || "demo";
```

```js
// Load graph data
const graphData = await loadGraph(requestId);
const intelligence = await loadIntelligence(requestId);
```

```js
// Extract nodes and edges from relationships
const relationships = graphData.relationships || [];
const nodes = new Map();
const edges = [];

relationships.forEach((rel, i) => {
  // Add source node
  if (!nodes.has(rel.source_id)) {
    nodes.set(rel.source_id, {
      id: rel.source_id,
      label: rel.source_title || rel.source_id,
      type: rel.source_type || "document",
      group: 1
    });
  }
  
  // Add target node
  if (!nodes.has(rel.target_id)) {
    nodes.set(rel.target_id, {
      id: rel.target_id,
      label: rel.target_title || rel.target_id,
      type: rel.target_type || "document",
      group: 2
    });
  }
  
  // Add edge
  edges.push({
    source: rel.source_id,
    target: rel.target_id,
    type: rel.relationship_type || "related",
    strength: rel.confidence || 0.5,
    value: rel.confidence || 0.5
  });
});

const graphNodes = Array.from(nodes.values());
```

```js
// Graph summary statistics
const graphStats = {
  nodes: graphNodes.length,
  edges: edges.length,
  relationships: relationships.length,
  avgConfidence: relationships.length > 0
    ? relationships.reduce((sum, r) => sum + (r.confidence || 0.5), 0) / relationships.length
    : 0
};
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
</div>

```js
import * as Plot from "@observablehq/plot";
```

```js
// Node type distribution
const nodeTypes = {};
graphNodes.forEach(node => {
  nodeTypes[node.type] = (nodeTypes[node.type] || 0) + 1;
});

const nodeTypeData = Object.entries(nodeTypes).map(([type, count]) => ({
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
      Plot.barY(nodeTypeData, {
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
const relTypes = {};
edges.forEach(edge => {
  relTypes[edge.type] = (relTypes[edge.type] || 0) + 1;
});

const relTypeData = Object.entries(relTypes).map(([type, count]) => ({
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
      Plot.barY(relTypeData, {
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
const confidenceData = relationships
  .map(r => r.confidence || 0.5)
  .filter(c => c > 0);
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
    {relationships.slice(0, 20).map((rel, i) => html`
      <div class="relationship-card">
        <div class="relationship-source">
          <strong>${rel.source_title || rel.source_id}</strong>
          <span class="relationship-type">${rel.relationship_type || "related"}</span>
        </div>
        <div class="relationship-arrow">→</div>
        <div class="relationship-target">
          <strong>${rel.target_title || rel.target_id}</strong>
        </div>
        <div class="relationship-confidence">
          ${((rel.confidence || 0.5) * 100).toFixed(1)}% confidence
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

