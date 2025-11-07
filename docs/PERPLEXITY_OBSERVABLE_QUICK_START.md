# Perplexity Observable Integration - Quick Start Guide

## Overview

This guide provides a quick reference for integrating Observable Plot, Runtime, Framework, and Stdlib into the Perplexity customer journey.

---

## Integration Summary

### Observable Framework
**Purpose**: Build interactive dashboards  
**Use Case**: Multi-page dashboard system for processing, results, analytics, and knowledge graphs

### Observable Plot
**Purpose**: Data visualization  
**Use Case**: Charts, graphs, network diagrams for all Perplexity data

### Observable Runtime
**Purpose**: Reactive UI  
**Use Case**: Real-time updates, auto-refresh, interactive filtering

### Observable Stdlib
**Purpose**: Rich presentation  
**Use Case**: HTML rendering, file downloads, DOM manipulation

---

## Quick Integration Checklist

### Phase 1: Setup
- [ ] Initialize Framework project: `npm create @observablehq/framework@latest dashboard`
- [ ] Install dependencies: `npm install @observablehq/plot @observablehq/runtime @observablehq/stdlib`
- [ ] Create data loaders for Perplexity API
- [ ] Set up API proxy for CORS

### Phase 2: Core Dashboards
- [ ] Processing Dashboard (status, progress, errors)
- [ ] Results Dashboard (documents, relationships, patterns)
- [ ] Analytics Dashboard (trends, metrics, distribution)

### Phase 3: Advanced Features
- [ ] Knowledge Graph Dashboard (interactive visualization)
- [ ] Query Dashboard (search interface)
- [ ] Real-time updates with Runtime
- [ ] Export functionality with Stdlib

---

## Key Integration Points

### 1. API Endpoints to Use

```javascript
// Processing Status
GET /api/perplexity/status/{request_id}

// Results
GET /api/perplexity/results/{request_id}

// Intelligence
GET /api/perplexity/results/{request_id}/intelligence

// Search Query
POST /api/perplexity/search

// Knowledge Graph Query
POST /api/perplexity/graph/{request_id}/query

// Domain Documents
GET /api/perplexity/domains/{domain}/documents

// Catalog Search
POST /api/perplexity/catalog/search
```

### 2. Data Loader Example

```javascript
// data/loaders/processing.js
export default async function(requestId) {
  const response = await fetch(`/api/perplexity/status/${requestId}`);
  if (!response.ok) throw new Error(`Failed to load: ${response.statusText}`);
  return response.json();
}
```

### 3. Visualization Example

```javascript
// src/components/charts.js
import * as Plot from "@observablehq/plot";

export function progressChart(requests) {
  return Plot.plot({
    marks: [
      Plot.line(requests, {
        x: "timestamp",
        y: "progress_percent",
        stroke: "request_id"
      })
    ],
    x: {label: "Time"},
    y: {label: "Progress %", domain: [0, 100]},
    width: 800
  });
}
```

### 4. Reactive UI Example

```javascript
// src/components/reactive.js
import {Runtime} from "@observablehq/runtime";

const runtime = new Runtime();
const module = runtime.module();

// Auto-refresh every 2 seconds
module.variable().define("status", async () => {
  const response = await fetch(`/api/perplexity/status/${requestId}`);
  return response.json();
});

// Reactive chart that updates automatically
module.variable().define("chart", ["status"], (status) => {
  return progressChart([status]);
});
```

### 5. Rich Presentation Example

```javascript
// src/components/cards.js
import {html, DOM} from "@observablehq/stdlib";

export function intelligenceCard(intelligence) {
  return html`
    <div class="card">
      <h3>Intelligence Summary</h3>
      <p>Domains: ${intelligence.domains.length}</p>
      <p>Relationships: ${intelligence.total_relationships}</p>
      <p>Patterns: ${intelligence.total_patterns}</p>
      ${DOM.download(
        () => new Blob([JSON.stringify(intelligence)], {type: "application/json"}),
        "intelligence.json"
      )}
    </div>
  `;
}
```

---

## Dashboard Structure

```
dashboard/
├── src/
│   ├── index.md              # Landing page with navigation
│   ├── processing.md         # Processing status dashboard
│   ├── results.md            # Results exploration dashboard
│   ├── analytics.md          # Analytics and trends dashboard
│   ├── graph.md              # Knowledge graph visualization
│   └── query.md              # Query interface dashboard
├── data/
│   └── loaders/
│       ├── processing.js     # Load processing status
│       ├── results.js        # Load results data
│       ├── intelligence.js   # Load intelligence data
│       └── analytics.js      # Load analytics data
└── observable.config.js      # Framework configuration
```

---

## Visualization Types

### Processing Dashboard
- **Progress Timeline**: Line chart of progress over time
- **Step Completion**: Bar chart of completed steps
- **Document Processing**: Stacked area chart
- **Error Rate**: Line chart of errors

### Results Dashboard
- **Domain Distribution**: Pie chart
- **Relationship Network**: Force-directed graph
- **Pattern Frequency**: Bar chart
- **Processing Time**: Histogram

### Analytics Dashboard
- **Request Volume**: Time series
- **Success Rate**: Line chart
- **Domain Distribution**: Treemap
- **Pattern Evolution**: Multi-line chart

### Knowledge Graph Dashboard
- **Graph Visualization**: Interactive network diagram
- **Node Distribution**: Bar chart
- **Edge Distribution**: Bar chart
- **Graph Metrics**: Summary cards

---

## Real-time Updates Pattern

```javascript
// Pattern for auto-refreshing data
import {Runtime} from "@observablehq/runtime";

const runtime = new Runtime();
const module = runtime.module();

function createAutoRefresh(fetchFn, interval = 2000) {
  let timeoutId;
  
  module.variable().define("data", async () => {
    const data = await fetchFn();
    
    // Schedule next refresh
    timeoutId = setTimeout(() => {
      module.redefine("data", async () => {
        return await fetchFn();
      });
    }, interval);
    
    return data;
  });
  
  return () => {
    if (timeoutId) clearTimeout(timeoutId);
  };
}
```

---

## Export Functionality

```javascript
import {DOM} from "@observablehq/stdlib";

// Export chart as PNG
export function exportChart(chartElement, filename = "chart.png") {
  return DOM.download(
    () => new Promise(resolve => {
      // Convert chart element to blob
      chartElement.toBlob(resolve, "image/png");
    }),
    filename
  );
}

// Export data as JSON
export function exportJSON(data, filename = "data.json") {
  return DOM.download(
    () => new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}),
    filename
  );
}

// Export data as CSV
export function exportCSV(data, filename = "data.csv") {
  const csv = convertToCSV(data);
  return DOM.download(
    () => new Blob([csv], {type: "text/csv"}),
    filename
  );
}
```

---

## API Integration Helper

```javascript
// utils/api.js
const API_BASE = "/api/perplexity";

export async function fetchStatus(requestId) {
  const response = await fetch(`${API_BASE}/status/${requestId}`);
  if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
  return response.json();
}

export async function fetchResults(requestId) {
  const response = await fetch(`${API_BASE}/results/${requestId}`);
  if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
  return response.json();
}

export async function fetchIntelligence(requestId) {
  const response = await fetch(`${API_BASE}/results/${requestId}/intelligence`);
  if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
  return response.json();
}

export async function searchQuery(query, topK = 10) {
  const response = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query, top_k: topK})
  });
  if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
  return response.json();
}
```

---

## Next Steps

1. **Review Integration Plan**: See `PERPLEXITY_OBSERVABLE_INTEGRATION_PLAN.md`
2. **Set up Development Environment**: Install Node.js, npm
3. **Initialize Framework Project**: Follow Phase 1 checklist
4. **Start with Processing Dashboard**: Build first visualization
5. **Iterate and Expand**: Add more dashboards and features

---

## Resources

- **Observable Framework**: https://observablehq.com/framework/
- **Observable Plot**: https://observablehq.com/plot/
- **Observable Runtime**: https://github.com/observablehq/runtime
- **Observable Stdlib**: https://github.com/observablehq/stdlib
- **Plot Gallery**: https://observablehq.com/@observablehq/plot-gallery

---

**Status**: 📋 Ready for Implementation

