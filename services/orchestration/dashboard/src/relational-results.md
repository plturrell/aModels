---
title: Relational Results Dashboard
description: Explore processed tables, relationships, and intelligence
---

```js
import * as Plot from "@observablehq/plot";
import relationalResultsData from "../data/loaders/relational-results.js";
import relationalIntelligenceData from "../data/loaders/relational-intelligence.js";
import {html} from "@observablehq/stdlib";
import {emptyStateNoRequest, emptyStateError} from "../components/emptyState.js";
```

# Relational Results Dashboard

<div class="fade-in">

Explore processed tables, discover relationships, and understand intelligence extracted from your relational data.

</div>

## Results Overview

```js
// Get request ID from URL parameters
const urlParams = typeof window !== "undefined" 
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();
const urlRequestId = urlParams.get("request_id") || "";

const requestId = typeof Inputs !== "undefined" && !urlRequestId
  ? await Inputs.text({label: "Request ID", value: urlRequestId, placeholder: "Enter request ID"})
  : urlRequestId;

let results = null;
let intelligence = null;
let resultsError = null;
let intelligenceError = null;

if (requestId) {
  try {
    results = await relationalResultsData(requestId);
    if (results && results.error) {
      resultsError = results;
      results = null;
    }
  } catch (error) {
    resultsError = { error: true, message: error.message || "Failed to load results" };
  }
  
  try {
    intelligence = await relationalIntelligenceData(requestId);
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

## Table Distribution by Domain

```js
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
  title: "Table Distribution by Domain",
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
    })
  ]
})
```

## Processed Tables

```js
html`<div class="card">
  <h3>Processed Tables (${results?.documents?.length || 0})</h3>
  ${resultsError ? emptyStateError(resultsError) : results && results.documents && results.documents.length > 0 ? html`
    <div style="margin-top: 16px;">
      ${results.documents.map(doc => html`
        <div style="padding: 16px; background: #F2F2F7; border-radius: 8px; margin-bottom: 12px;">
          <div style="font-size: 18px; font-weight: 600; color: #1d1d1f; margin-bottom: 4px;">
            ${doc.id}
          </div>
          <div style="font-size: 14px; color: #86868b;">
            Status: ${doc.status} • Processed: ${doc.processed_at}
          </div>
        </div>
      `)}
    </div>
  ` : emptyStateNoRequest()}
</div>`
```

