---
title: Query
description: Visual query builder and search interface
---

# Query Dashboard

<div class="dashboard-header">
  <h1>Query Interface</h1>
  <p>Search and explore processed documents with visual query builder</p>
</div>

```js
import {html, DOM} from "@observablehq/stdlib";
```

```js
// Query state
let queryText = "";
let searchResults = [];
let searchLoading = false;
let searchError = null;
```

```js
// Search function
async function performSearch(query) {
  searchLoading = true;
  searchError = null;
  
  try {
    const apiBase = process.env.PERPLEXITY_API_BASE || "http://localhost:8080";
    const response = await fetch(`${apiBase}/api/perplexity/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 20 })
    });
    
    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    searchResults = data.results || [];
  } catch (error) {
    searchError = error.message;
    searchResults = [];
  } finally {
    searchLoading = false;
  }
  
  return searchResults;
}
```

<div class="query-builder">
  <div class="query-input-group">
    <label for="query-input">Search Query</label>
    <input 
      id="query-input"
      type="text" 
      placeholder="Enter your search query..."
      value=${queryText}
      oninput=${(e) => { queryText = e.target.value; }}
      style="
        width: 100%;
        padding: 12px 16px;
        font-size: 16px;
        border: 1px solid #E5E5EA;
        border-radius: 8px;
        margin-top: 8px;
      "
    />
  </div>
  
  <button 
    onclick=${async () => {
      if (queryText.trim()) {
        await performSearch(queryText);
        DOM.redraw();
      }
    }}
    disabled=${searchLoading}
    style="
      padding: 12px 24px;
      font-size: 16px;
      font-weight: 500;
      background: #007AFF;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      margin-top: 16px;
      opacity: ${searchLoading ? 0.6 : 1};
    "
  >
    ${searchLoading ? "Searching..." : "Search"}
  </button>
</div>

```js
// Search performance metrics
const searchMetrics = {
  totalResults: searchResults.length,
  avgScore: searchResults.length > 0
    ? searchResults.reduce((sum, r) => sum + (r.score || r.similarity || 0), 0) / searchResults.length
    : 0,
  maxScore: searchResults.length > 0
    ? Math.max(...searchResults.map(r => r.score || r.similarity || 0))
    : 0
};
```

${searchResults.length > 0 ? html`
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value">${searchMetrics.totalResults}</div>
      <div class="stat-label">Results</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${(searchMetrics.avgScore * 100).toFixed(1)}%</div>
      <div class="stat-label">Avg Relevance</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${(searchMetrics.maxScore * 100).toFixed(1)}%</div>
      <div class="stat-label">Max Relevance</div>
    </div>
  </div>
` : ""}

${searchError ? html`
  <div class="error-message">
    <strong>Error:</strong> ${searchError}
  </div>
` : ""}

```js
import * as Plot from "@observablehq/plot";
```

```js
// Relevance score distribution
const scoreData = searchResults.map((r, i) => ({
  index: i,
  score: r.score || r.similarity || 0,
  title: r.title || r.id || `Result ${i + 1}`
}));
```

${searchResults.length > 0 ? html`
  <div class="chart-container">
    <h2>Relevance Scores</h2>
    ${Plot.plot({
      margin: {top: 20, right: 20, bottom: 60, left: 60},
      width: 800,
      height: 400,
      marks: [
        Plot.dot(scoreData, {
          x: "index",
          y: "score",
          fill: "#007AFF",
          fillOpacity: 0.6,
          r: 4
        }),
        Plot.ruleY([0, 1], {
          stroke: "#E5E5EA",
          strokeWidth: 1
        })
      ],
      x: {
        label: "Result Index",
        labelFontSize: 14,
        labelFontWeight: "500"
      },
      y: {
        label: "Relevance Score",
        labelFontSize: 14,
        labelFontWeight: "500",
        domain: [0, 1]
      }
    })}
  </div>
` : ""}

```js
// Score histogram
const scoreBins = Array.from({length: 10}, (_, i) => ({
  bin: i / 10,
  count: scoreData.filter(d => d.score >= i / 10 && d.score < (i + 1) / 10).length
}));
```

${searchResults.length > 0 ? html`
  <div class="chart-container">
    <h2>Score Distribution</h2>
    ${Plot.plot({
      margin: {top: 20, right: 20, bottom: 60, left: 60},
      width: 800,
      height: 400,
      marks: [
        Plot.barY(scoreBins, {
          x: "bin",
          y: "count",
          fill: "#34C759",
          fillOpacity: 0.8
        })
      ],
      x: {
        label: "Score Range",
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
` : ""}

${searchResults.length > 0 ? html`
  <div class="results-list">
    <h2>Search Results</h2>
    <div class="results-grid">
      ${searchResults.map((result, i) => html`
        <div class="result-card">
          <div class="result-header">
            <span class="result-rank">#${i + 1}</span>
            <span class="result-score">${((result.score || result.similarity || 0) * 100).toFixed(1)}%</span>
          </div>
          <h3 class="result-title">${result.title || result.id || "Untitled"}</h3>
          <p class="result-content">${(result.content || result.text || "").substring(0, 200)}...</p>
          ${result.metadata ? html`
            <div class="result-metadata">
              ${Object.entries(result.metadata).slice(0, 3).map(([key, value]) => html`
                <span class="metadata-tag">${key}: ${value}</span>
              `)}
            </div>
          ` : ""}
        </div>
      `)}
    </div>
  </div>
` : html`
  <div class="empty-state">
    <p>Enter a search query above to find documents</p>
  </div>
`}

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

.query-builder {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  margin-bottom: 32px;
}

.query-input-group label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #1d1d1f;
  margin-bottom: 8px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
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

.error-message {
  background: #FF3B30;
  color: white;
  padding: 16px;
  border-radius: 8px;
  margin: 16px 0;
}

.results-list {
  margin: 32px 0;
}

.results-list h2 {
  font-size: 24px;
  font-weight: 600;
  color: #1d1d1f;
  margin: 0 0 20px 0;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.result-card {
  background: white;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.result-rank {
  font-size: 12px;
  color: #86868b;
  font-weight: 500;
}

.result-score {
  font-size: 12px;
  color: #007AFF;
  font-weight: 600;
  background: #F2F2F7;
  padding: 4px 8px;
  border-radius: 6px;
}

.result-title {
  font-size: 16px;
  font-weight: 600;
  color: #1d1d1f;
  margin: 0 0 8px 0;
}

.result-content {
  font-size: 14px;
  color: #86868b;
  line-height: 1.5;
  margin: 0 0 12px 0;
}

.result-metadata {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.metadata-tag {
  font-size: 12px;
  color: #86868b;
  background: #F2F2F7;
  padding: 4px 8px;
  border-radius: 6px;
}

.empty-state {
  text-align: center;
  padding: 64px;
  color: #86868b;
  font-size: 16px;
}
</style>

