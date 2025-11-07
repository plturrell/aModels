---
title: Results Dashboard
description: Explore processed documents, relationships, and intelligence
---

```js
import * as Plot from "@observablehq/plot";
import resultsData from "../data/loaders/results.js";
import intelligenceData from "../data/loaders/intelligence.js";
import {html, DOM} from "@observablehq/stdlib";
import {exportJSON, exportCSV, exportChartPNG, exportChartSVG} from "../components/export.js";
import {emptyStateNoRequest, emptyStateNoData, emptyStateError} from "../components/emptyState.js";
```

# Results Dashboard

<div class="fade-in">

Explore processed documents, visualize relationships, and discover patterns in your data.

</div>

## Results Overview

```js
// Get request ID from URL parameters (deep linking support)
const urlParams = typeof window !== "undefined" 
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();
const urlRequestId = urlParams.get("request_id") || "";

// Input for request ID (if not in URL)
const requestId = typeof Inputs !== "undefined" && !urlRequestId
  ? await Inputs.text({label: "Request ID", value: urlRequestId, placeholder: "Enter request ID or use ?request_id=xxx in URL"})
  : urlRequestId;

// Load data with error handling
let results = null;
let intelligence = null;
let resultsError = null;
let intelligenceError = null;

if (requestId) {
  try {
    results = await resultsData(requestId);
    if (results && results.error) {
      resultsError = results;
      results = null;
    }
  } catch (error) {
    resultsError = { error: true, message: error.message || "Failed to load results" };
  }
  
  try {
    intelligence = await intelligenceData(requestId);
    if (intelligence && intelligence.error) {
      intelligenceError = intelligence;
      intelligence = null;
    }
  } catch (error) {
    intelligenceError = { error: true, message: error.message || "Failed to load intelligence" };
  }
}
```

```js
// Intelligence Summary Card
html`<div class="card">
  ${intelligenceError ? emptyStateError(intelligenceError) : intelligence ? html`
    <h3>Intelligence Summary</h3>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-top: 24px;">
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
          ${intelligence.intelligence?.domains?.length || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Domains</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
          ${intelligence.intelligence?.total_relationships || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Relationships</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
          ${intelligence.intelligence?.total_patterns || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Patterns</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
          ${intelligence.intelligence?.knowledge_graph_nodes || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">KG Nodes</div>
      </div>
    </div>
  ` : emptyStateNoRequest()}
</div>`
```

## Document Distribution by Domain

```js
// Domain Distribution Pie Chart
// Design: Simple, clear, beautiful
function domainDistributionChart(intelligence) {
  if (!intelligence || !intelligence.intelligence?.domains) return null;
  
  const domains = intelligence.intelligence.domains;
  const domainCounts = {};
  
  // Count documents per domain
  if (intelligence.documents) {
    intelligence.documents.forEach(doc => {
      const domain = doc.intelligence?.domain || "Unknown";
      domainCounts[domain] = (domainCounts[domain] || 0) + 1;
    });
  }
  
  const data = Object.entries(domainCounts).map(([domain, count], i) => ({
    domain,
    count,
    color: ["#007AFF", "#34C759", "#FF9500", "#AF52DE", "#FF2D55", "#5AC8FA"][i % 6]
  }));
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 20, left: 20},
    
    marks: [
      Plot.arc(data, {
        innerRadius: 0,
        outerRadius: 150,
        padAngle: 0.02,
        fill: "color",
        stroke: "white",
        strokeWidth: 2
      }),
      Plot.text(data, {
        x: 0,
        y: 0,
        text: (d) => `${d.domain}\n${d.count}`,
        fontSize: 12,
        fontWeight: "500",
        fill: "#1d1d1f"
      })
    ],
    
    width: 400,
    height: 400,
    
    style: {
      background: "transparent"
    }
  });
}

intelligence ? domainDistributionChart(intelligence) : null
```

## Relationship Network

```js
// Relationship Network Visualization
// Design: Organic, flowing, beautiful
function relationshipNetworkChart(intelligence) {
  if (!intelligence || !intelligence.intelligence?.total_relationships) return null;
  
  // Extract relationships from documents
  const relationships = [];
  if (intelligence.documents) {
    intelligence.documents.forEach(doc => {
      if (doc.intelligence?.relationships) {
        doc.intelligence.relationships.forEach(rel => {
          relationships.push({
            source: rel.source || "Unknown",
            target: rel.target || "Unknown",
            type: rel.type || "related",
            strength: rel.strength || 1
          });
        });
      }
    });
  }
  
  if (relationships.length === 0) return null;
  
  // Create nodes
  const nodes = new Set();
  relationships.forEach(rel => {
    nodes.add(rel.source);
    nodes.add(rel.target);
  });
  
  const nodeData = Array.from(nodes).map((node, i) => ({
    id: node,
    x: Math.cos((i / nodes.size) * 2 * Math.PI) * 150 + 200,
    y: Math.sin((i / nodes.size) * 2 * Math.PI) * 150 + 200,
    size: 8
  }));
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 20, left: 20},
    
    marks: [
      // Edges
      Plot.link(relationships, {
        x1: (d) => nodeData.find(n => n.id === d.source)?.x || 200,
        y1: (d) => nodeData.find(n => n.id === d.source)?.y || 200,
        x2: (d) => nodeData.find(n => n.id === d.target)?.x || 200,
        y2: (d) => nodeData.find(n => n.id === d.target)?.y || 200,
        stroke: "#007AFF",
        strokeWidth: 1,
        strokeOpacity: 0.3
      }),
      // Nodes
      Plot.dot(nodeData, {
        x: "x",
        y: "y",
        r: "size",
        fill: "#007AFF",
        stroke: "white",
        strokeWidth: 2
      }),
      // Labels
      Plot.text(nodeData, {
        x: "x",
        y: "y",
        text: "id",
        fontSize: 10,
        dx: 10,
        dy: 5,
        fill: "#1d1d1f"
      })
    ],
    
    width: 600,
    height: 600,
    
    style: {
      background: "transparent"
    }
  });
}

intelligence ? relationshipNetworkChart(intelligence) : null
```

## Pattern Frequency

```js
// Pattern Frequency Bar Chart
// Design: Clean, easy to scan
function patternFrequencyChart(intelligence) {
  if (!intelligence || !intelligence.intelligence?.total_patterns) return null;
  
  // Extract patterns from documents
  const patterns = {};
  if (intelligence.documents) {
    intelligence.documents.forEach(doc => {
      if (doc.intelligence?.learned_patterns) {
        doc.intelligence.learned_patterns.forEach(pattern => {
          const key = pattern.type || pattern.name || "Unknown";
          patterns[key] = (patterns[key] || 0) + 1;
        });
      }
    });
  }
  
  if (Object.keys(patterns).length === 0) return null;
  
  const data = Object.entries(patterns)
    .map(([name, count]) => ({name, count}))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10); // Top 10
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 60, left: 60},
    
    marks: [
      Plot.barX(data, {
        y: "name",
        x: "count",
        fill: "#007AFF",
        rx: 4
      }),
      Plot.ruleX([0])
    ],
    
    x: {
      label: "Frequency",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    y: {
      label: null,
      labelFontSize: 12
    },
    
    width: 600,
    height: Math.max(300, data.length * 30)
  });
}

intelligence ? patternFrequencyChart(intelligence) : null
```

## Processing Time Distribution

```js
// Processing Time Histogram
// Design: Clear distribution, obvious patterns
function processingTimeChart(results) {
  if (!results || !results.documents) return null;
  
  const processingTimes = results.documents
    .map(doc => {
      if (doc.processed_at && results.request?.created_at) {
        const start = new Date(results.request.created_at);
        const end = new Date(doc.processed_at);
        return (end - start) / 1000; // seconds
      }
      return null;
    })
    .filter(t => t !== null);
  
  if (processingTimes.length === 0) return null;
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.rectY(processingTimes, Plot.binX({y: "count"}, {x: "value", fill: "#007AFF", fillOpacity: 0.7})),
      Plot.ruleY([0])
    ],
    
    x: {
      label: "Processing Time (seconds)",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    y: {
      label: "Count",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    
    width: 600,
    height: 300
  });
}

results ? processingTimeChart(results) : null
```

## Documents List

```js
// Documents List
html`<div class="card">
  <h3>Processed Documents</h3>
  ${results && results.documents ? html`
    <div style="margin-top: 16px;">
      ${results.documents.map(doc => html`
        <div style="padding: 16px; border-bottom: 1px solid #E5E5EA; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-size: 16px; font-weight: 500; color: #1d1d1f; margin-bottom: 4px;">
              ${doc.title || doc.id}
            </div>
            <div style="font-size: 14px; color: #86868b;">
              ${doc.intelligence?.domain || "Unknown domain"} • 
              ${doc.status === "success" ? "✓ Success" : "✗ Failed"}
            </div>
          </div>
          <div style="font-size: 14px; color: #86868b;">
            ${doc.intelligence?.total_relationships || 0} relationships
          </div>
        </div>
      `)}
    </div>
  ` : emptyStateNoData("No documents available")}
</div>`
```

---

*Explore your data with beautiful, intuitive visualizations.*

