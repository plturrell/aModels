---
title: Analytics Dashboard
description: Analyze trends, patterns, and domain distributions
---

```js
import * as Plot from "@observablehq/plot";
import analyticsData from "../data/loaders/analytics.js";
import {html} from "@observablehq/stdlib";
```

# Analytics Dashboard

<div class="fade-in">

Analyze trends, discover patterns, and understand your data processing performance.

</div>

## Analytics Overview

```js
// Load analytics data
const analytics = await analyticsData({limit: 100, offset: 0});
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
      <div style="font-size: 32px; font-weight: 600; color: #FF9500; margin-bottom: 4px;">
        ${analytics.requests?.filter(r => r.status === 'processing').length || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Processing</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 32px; font-weight: 600; color: #FF3B30; margin-bottom: 4px;">
        ${analytics.requests?.filter(r => r.status === 'failed').length || 0}
      </div>
      <div style="font-size: 14px; color: #86868b;">Failed</div>
    </div>
  </div>
</div>`
```

## Request Volume Over Time

```js
// Request Volume Time Series
// Design: Elegant, smooth curves, clear trends
function requestVolumeChart(analytics) {
  if (!analytics || !analytics.trends || analytics.trends.length === 0) return null;
  
  const data = analytics.trends.map(t => ({
    date: new Date(t.date),
    count: t.count,
    success: t.success,
    failed: t.failed
  }));
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.areaY(data, {
        x: "date",
        y: "count",
        fill: "#007AFF",
        fillOpacity: 0.2,
        curve: "natural"
      }),
      Plot.line(data, {
        x: "date",
        y: "count",
        stroke: "#007AFF",
        strokeWidth: 2,
        curve: "natural"
      }),
      Plot.line(data, {
        x: "date",
        y: "success",
        stroke: "#34C759",
        strokeWidth: 2,
        curve: "natural"
      }),
      Plot.line(data, {
        x: "date",
        y: "failed",
        stroke: "#FF3B30",
        strokeWidth: 2,
        curve: "natural"
      })
    ],
    
    x: {
      label: "Date",
      labelFontSize: 14,
      labelFontWeight: "500",
      tickFormat: (d) => new Date(d).toLocaleDateString()
    },
    y: {
      label: "Requests",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    
    width: 800,
    height: 400,
    
    color: {
      legend: true
    }
  });
}

analytics ? requestVolumeChart(analytics) : null
```

## Success Rate Trends

```js
// Success Rate Over Time
// Design: Green = good, obvious at a glance
function successRateChart(analytics) {
  if (!analytics || !analytics.trends || analytics.trends.length === 0) return null;
  
  const data = analytics.trends.map(t => ({
    date: new Date(t.date),
    successRate: t.count > 0 ? (t.success / t.count) * 100 : 0,
    count: t.count
  }));
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.areaY(data, {
        x: "date",
        y: "successRate",
        fill: "#34C759",
        fillOpacity: 0.3,
        curve: "natural"
      }),
      Plot.line(data, {
        x: "date",
        y: "successRate",
        stroke: "#34C759",
        strokeWidth: 2,
        curve: "natural"
      }),
      Plot.ruleY([90], {
        stroke: "#FF9500",
        strokeWidth: 1,
        strokeDasharray: "4,4"
      })
    ],
    
    x: {
      label: "Date",
      labelFontSize: 14,
      labelFontWeight: "500",
      tickFormat: (d) => new Date(d).toLocaleDateString()
    },
    y: {
      label: "Success Rate (%)",
      labelFontSize: 14,
      labelFontWeight: "500",
      domain: [0, 100],
      tickFormat: (d) => `${d}%`
    },
    
    width: 800,
    height: 400
  });
}

analytics ? successRateChart(analytics) : null
```

## Domain Distribution Treemap

```js
// Domain Distribution Treemap
// Design: Color with purpose, clear hierarchy
function domainTreemapChart(analytics) {
  if (!analytics || !analytics.requests) return null;
  
  // Extract domain distribution from requests
  const domainCounts = {};
  analytics.requests.forEach(req => {
    if (req.intelligence?.domains) {
      req.intelligence.domains.forEach(domain => {
        domainCounts[domain] = (domainCounts[domain] || 0) + 1;
      });
    }
  });
  
  if (Object.keys(domainCounts).length === 0) return null;
  
  const data = Object.entries(domainCounts)
    .map(([domain, count]) => ({
      domain,
      count,
      value: count
    }))
    .sort((a, b) => b.count - a.count);
  
  // Simple treemap using rectangles
  const colors = ["#007AFF", "#34C759", "#FF9500", "#AF52DE", "#FF2D55", "#5AC8FA"];
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 20},
    
    marks: [
      Plot.cell(data, {
        x: (d, i) => (i % 3) * 200 + 100,
        y: (d, i) => Math.floor(i / 3) * 100 + 50,
        width: (d) => Math.sqrt(d.value) * 20,
        height: (d) => Math.sqrt(d.value) * 20,
        fill: (d, i) => colors[i % colors.length],
        fillOpacity: 0.7,
        stroke: "white",
        strokeWidth: 2
      }),
      Plot.text(data, {
        x: (d, i) => (i % 3) * 200 + 100,
        y: (d, i) => Math.floor(i / 3) * 100 + 50,
        text: (d) => `${d.domain}\n${d.count}`,
        fontSize: 12,
        fontWeight: "500",
        fill: "#1d1d1f"
      })
    ],
    
    width: 800,
    height: Math.max(400, Math.ceil(data.length / 3) * 120),
    
    style: {
      background: "transparent"
    }
  });
}

analytics ? domainTreemapChart(analytics) : null
```

## Processing Performance

```js
// Processing Performance Metrics
// Design: Clear metrics, obvious what's good/bad
function performanceChart(analytics) {
  if (!analytics || !analytics.requests) return null;
  
  const completed = analytics.requests.filter(r => r.status === 'completed');
  if (completed.length === 0) return null;
  
  const processingTimes = completed
    .map(r => {
      if (r.completed_at && r.created_at) {
        const start = new Date(r.created_at);
        const end = new Date(r.completed_at);
        return (end - start) / 1000; // seconds
      }
      return null;
    })
    .filter(t => t !== null);
  
  if (processingTimes.length === 0) return null;
  
  const avgTime = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
  const minTime = Math.min(...processingTimes);
  const maxTime = Math.max(...processingTimes);
  
  const data = [
    {metric: "Average", value: avgTime, color: "#007AFF"},
    {metric: "Minimum", value: minTime, color: "#34C759"},
    {metric: "Maximum", value: maxTime, color: "#FF9500"}
  ];
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.barY(data, {
        x: "metric",
        y: "value",
        fill: "color",
        rx: 4
      }),
      Plot.ruleY([0])
    ],
    
    x: {
      label: null,
      labelFontSize: 14
    },
    y: {
      label: "Time (seconds)",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    
    width: 400,
    height: 300
  });
}

analytics ? performanceChart(analytics) : null
```

## Recent Activity

```js
// Recent Activity List
html`<div class="card">
  <h3>Recent Activity</h3>
  ${analytics && analytics.requests ? html`
    <div style="margin-top: 16px;">
      ${analytics.requests.slice(0, 10).map(req => html`
        <div style="padding: 16px; border-bottom: 1px solid #E5E5EA; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-size: 16px; font-weight: 500; color: #1d1d1f; margin-bottom: 4px;">
              ${req.query || req.request_id}
            </div>
            <div style="font-size: 14px; color: #86868b;">
              ${new Date(req.created_at).toLocaleString()} • 
              ${req.statistics?.documents_processed || 0} documents
            </div>
          </div>
          <div style="
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            background: ${req.status === 'completed' ? '#E8F5E9' : req.status === 'processing' ? '#FFF3E0' : '#FFEBEE'};
            color: ${req.status === 'completed' ? '#2E7D32' : req.status === 'processing' ? '#F57C00' : '#C62828'};
          ">
            ${req.status}
          </div>
        </div>
      `)}
    </div>
  ` : html`
    <div class="empty-state">
      <p>No activity data available</p>
    </div>
  `}
</div>`
```

---

*Discover insights with beautiful, intuitive analytics.*

