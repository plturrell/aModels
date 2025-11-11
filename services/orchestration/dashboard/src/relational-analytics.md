---
title: Relational Analytics Dashboard
description: Analyze trends, patterns, and domain distributions
---

```js
import * as Plot from "@observablehq/plot";
import relationalAnalyticsData from "../data/loaders/relational-analytics.js";
import {html} from "@observablehq/stdlib";
```

# Relational Analytics Dashboard

<div class="fade-in">

Analyze trends, discover patterns, and understand your relational table processing performance.

</div>

## Analytics Overview

```js
const analytics = await relationalAnalyticsData();
```

```js
html`<div class="card">
  <h3>Analytics Summary</h3>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-top: 24px;">
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
        ${analytics.trends?.total || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Total Requests</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #34C759; margin-bottom: 4px;">
        ${analytics.trends?.completed || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Completed</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #FF3B30; margin-bottom: 4px;">
        ${analytics.trends?.failed || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Failed</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #FF9500; margin-bottom: 4px;">
        ${analytics.trends?.success_rate?.toFixed(1) || 0}%
      </div>
      <div style="font-size: 14px; color: #86868b;">Success Rate</div>
    </div>
  </div>
</div>`
```

## Request Volume Over Time

```js
const volumeData = analytics.requests ? analytics.requests
  .filter(r => r.created_at)
  .map(r => ({
    date: new Date(r.created_at),
    count: 1
  }))
  .reduce((acc, r) => {
    const date = r.date.toISOString().split('T')[0];
    acc[date] = (acc[date] || 0) + 1;
    return acc;
  }, {}) : {};

const volumeEntries = Object.entries(volumeData).map(([date, count]) => ({
  date: new Date(date),
  count
})).sort((a, b) => a.date - b.date);

Plot.plot({
  title: "Request Volume Over Time",
  width: 800,
  height: 300,
  marginLeft: 60,
  x: {label: "Date", type: "time"},
  y: {label: "Count"},
  marks: [
    Plot.areaY(volumeEntries, {
      x: "date",
      y: "count",
      fill: "#007AFF",
      fillOpacity: 0.3
    }),
    Plot.lineY(volumeEntries, {
      x: "date",
      y: "count",
      stroke: "#007AFF",
      strokeWidth: 2
    })
  ]
})
```

## Recent Activity

```js
const recentRequests = analytics.requests ? analytics.requests
  .filter(r => r.created_at)
  .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
  .slice(0, 10) : [];

html`<div class="card">
  <h3>Recent Activity</h3>
  ${recentRequests.length > 0 ? html`
    <div style="margin-top: 16px;">
      ${recentRequests.map(req => html`
        <div style="padding: 12px; border-bottom: 1px solid #E5E5EA;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-size: 14px; font-weight: 600; color: #1d1d1f;">
                ${req.request_id}
              </div>
              <div style="font-size: 12px; color: #86868b; margin-top: 4px;">
                ${req.query || 'No query'}
              </div>
            </div>
            <div style="padding: 4px 12px; background: ${req.status === 'completed' ? '#34C759' : req.status === 'failed' ? '#FF3B30' : '#FF9500'}; color: white; border-radius: 12px; font-size: 12px;">
              ${req.status}
            </div>
          </div>
        </div>
      `)}
    </div>
  ` : html`
    <div style="text-align: center; padding: 32px; color: #86868b;">
      No recent activity
    </div>
  `}
</div>`
```

