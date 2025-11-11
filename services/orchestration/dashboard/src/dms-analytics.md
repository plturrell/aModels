---
title: DMS Analytics Dashboard
description: Analyze trends, patterns, and domain distributions
---

```js
import * as Plot from "@observablehq/plot";
import dmsAnalyticsData from "../data/loaders/dms-analytics.js";
import {html} from "@observablehq/stdlib";
```

# DMS Analytics Dashboard

<div class="fade-in">

Analyze trends, discover patterns, and understand your document processing performance.

</div>

## Analytics Overview

```js
// Load analytics data
const analytics = await dmsAnalyticsData({limit: 100, offset: 0});
```

```js
// Analytics Summary Card
html`<div class="card">
  <h3>Analytics Summary</h3>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-top: 24px;">
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
        ${analytics.total || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Total Requests</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #34C759; margin-bottom: 4px;">
        ${analytics.requests?.filter(r => r.status === 'completed').length || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Completed</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #FF3B30; margin-bottom: 4px;">
        ${analytics.requests?.filter(r => r.status === 'failed').length || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Failed</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #FF9500; margin-bottom: 4px;">
        ${analytics.requests?.filter(r => r.status === 'completed').length > 0 
          ? ((analytics.requests.filter(r => r.status === 'completed').length / analytics.total) * 100).toFixed(1) 
          : 0}%
      </div>
      <div style="font-size: 14px; color: #86868b;">Success Rate</div>
    </div>
  </div>
</div>`
```

## Request Volume Time Series

```js
// Request volume over time
const volumeData = analytics.trends || [];

Plot.plot({
  title: "Request Volume Over Time",
  width: 800,
  height: 300,
  marginLeft: 60,
  x: {label: "Date", type: "time"},
  y: {label: "Count"},
  marks: [
    Plot.areaY(volumeData, {
      x: d => new Date(d.date),
      y: "count",
      fill: "#007AFF",
      fillOpacity: 0.3
    }),
    Plot.lineY(volumeData, {
      x: d => new Date(d.date),
      y: "count",
      stroke: "#007AFF",
      strokeWidth: 2
    })
  ]
})
```

## Success Rate Trends

```js
// Success rate over time
const successData = analytics.trends || [];

Plot.plot({
  title: "Success Rate Trends",
  width: 800,
  height: 300,
  marginLeft: 60,
  x: {label: "Date", type: "time"},
  y: {label: "Rate", domain: [0, 100]},
  marks: [
    Plot.areaY(successData, {
      x: d => new Date(d.date),
      y: d => d.count > 0 ? (d.success / d.count) * 100 : 0,
      fill: "#34C759",
      fillOpacity: 0.3
    }),
    Plot.lineY(successData, {
      x: d => new Date(d.date),
      y: d => d.count > 0 ? (d.success / d.count) * 100 : 0,
      stroke: "#34C759",
      strokeWidth: 2
    })
  ]
})
```

## Recent Activity

```js
html`<div class="card">
  <h3>Recent Requests</h3>
  ${analytics.requests && analytics.requests.length > 0 ? html`
    <div style="margin-top: 16px;">
      ${analytics.requests.slice(0, 10).map(req => html`
        <div style="padding: 12px; background: #F2F2F7; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-size: 14px; font-weight: 500; color: #1d1d1f;">
              ${req.request_id}
            </div>
            <div style="font-size: 12px; color: #86868b; margin-top: 4px;">
              ${new Date(req.created_at).toLocaleString()}
            </div>
          </div>
          <div style="padding: 4px 12px; background: ${req.status === 'completed' ? '#34C759' : req.status === 'failed' ? '#FF3B30' : '#FF9500'}; color: white; border-radius: 12px; font-size: 12px; font-weight: 500;">
            ${req.status}
          </div>
        </div>
      `)}
    </div>
  ` : html`
    <div style="text-align: center; padding: 32px; color: #86868b;">
      No requests found
    </div>
  `}
</div>`
```

