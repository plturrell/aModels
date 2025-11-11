---
title: DMS Results Dashboard
description: Explore processed documents, relationships, and intelligence
---

```js
import * as Plot from "@observablehq/plot";
import dmsResultsData from "../data/loaders/dms-results.js";
import dmsIntelligenceData from "../data/loaders/dms-intelligence.js";
import {html, DOM} from "@observablehq/stdlib";
import {exportJSON, exportCSV, exportChartPNG, exportChartSVG} from "../components/export.js";
import {emptyStateNoRequest, emptyStateNoData, emptyStateError} from "../components/emptyState.js";
```

# DMS Results Dashboard

<div class="fade-in">

Explore processed documents, discover relationships, and understand intelligence extracted from your documents.

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
    results = await dmsResultsData(requestId);
    if (results && results.error) {
      resultsError = results;
      results = null;
    }
  } catch (error) {
    resultsError = { error: true, message: error.message || "Failed to load results" };
  }
  
  try {
    intelligence = await dmsIntelligenceData(requestId);
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
        <div style="font-size: 32px; font-weight: 600; color: #34C759; margin-bottom: 4px;">
          ${intelligence.intelligence?.total_relationships || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Relationships</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #FF9500; margin-bottom: 4px;">
          ${intelligence.intelligence?.total_patterns || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Patterns</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #5856D6; margin-bottom: 4px;">
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
// Domain distribution pie chart
const domainData = results && results.documents ? 
  results.documents
    .filter(doc => doc.intelligence?.domain)
    .reduce((acc, doc) => {
      const domain = doc.intelligence.domain;
      acc[domain] = (acc[domain] || 0) + 1;
      return acc;
    }, {}) : {};

const domainEntries = Object.entries(domainData).map(([domain, count]) => ({
  domain,
  count
}));

Plot.plot({
  title: "Document Distribution by Domain",
  width: 500,
  height: 500,
  color: {scheme: "Tableau10"},
  marks: [
    Plot.arc(domainEntries, {
      innerRadius: 80,
      outerRadius: 200,
      fill: "domain",
      padAngle: 0.02,
      tip: true
    }),
    Plot.text(domainEntries, {
      innerRadius: 140,
      text: d => `${d.domain}\n${d.count}`,
      fontSize: 12,
      fontWeight: 500
    })
  ]
})
```

## Relationship Network

```js
// Relationship network visualization
const relationships = intelligence && intelligence.intelligence ? 
  (intelligence.documents || []).flatMap(doc => 
    (doc.intelligence?.relationships || []).map(rel => ({
      source: doc.id,
      target: rel.target_id || rel.target_title,
      type: rel.type,
      strength: rel.strength || 0.5
    }))
  ) : [];

// Simple relationship visualization
html`<div class="card">
  <h3>Relationships</h3>
  ${relationships.length > 0 ? html`
    <div style="margin-top: 16px;">
      ${relationships.slice(0, 10).map(rel => html`
        <div style="padding: 12px; background: #F2F2F7; border-radius: 8px; margin-bottom: 8px;">
          <div style="font-size: 14px; font-weight: 500; color: #1d1d1f;">
            ${rel.source} → ${rel.target}
          </div>
          <div style="font-size: 12px; color: #86868b; margin-top: 4px;">
            ${rel.type} (strength: ${(rel.strength * 100).toFixed(0)}%)
          </div>
        </div>
      `)}
    </div>
  ` : html`
    <div style="text-align: center; padding: 32px; color: #86868b;">
      No relationships found
    </div>
  `}
</div>`
```

## Pattern Frequency

```js
// Pattern frequency bar chart
const patterns = intelligence && intelligence.intelligence ?
  (intelligence.documents || []).flatMap(doc => 
    (doc.intelligence?.learned_patterns || []).map(p => p.type)
  ) : [];

const patternCounts = patterns.reduce((acc, type) => {
  acc[type] = (acc[type] || 0) + 1;
  return acc;
}, {});

const patternData = Object.entries(patternCounts).map(([type, count]) => ({
  type,
  count
})).sort((a, b) => b.count - a.count);

Plot.plot({
  title: "Pattern Frequency",
  width: 600,
  height: 300,
  marginLeft: 120,
  y: {label: "Pattern Type"},
  x: {label: "Frequency"},
  marks: [
    Plot.barX(patternData, {
      x: "count",
      y: "type",
      fill: "#007AFF"
    })
  ]
})
```

## Documents List

```js
html`<div class="card">
  <h3>Processed Documents</h3>
  ${results && results.documents && results.documents.length > 0 ? html`
    <div style="margin-top: 16px;">
      ${results.documents.map(doc => html`
        <div style="padding: 16px; background: #F2F2F7; border-radius: 8px; margin-bottom: 12px;">
          <div style="font-size: 18px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px;">
            ${doc.title || doc.id}
          </div>
          <div style="font-size: 14px; color: #86868b; margin-bottom: 8px;">
            ${doc.intelligence?.domain ? `Domain: ${doc.intelligence.domain}` : ''}
          </div>
          <div style="font-size: 12px; color: #86868b;">
            ${doc.intelligence?.relationships?.length || 0} relationships •
            ${doc.intelligence?.learned_patterns?.length || 0} patterns
          </div>
        </div>
      `)}
    </div>
  ` : emptyStateNoData("No documents available")}
</div>`
```

