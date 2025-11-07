---
title: Processing Dashboard
description: Real-time processing status, progress, and statistics
---

```js
import * as Plot from "@observablehq/plot";
import processingStatus from "../data/loaders/processing.js";
import {html} from "@observablehq/stdlib";
import {emptyStateNoRequest, emptyStateLoading, emptyStateError} from "../components/emptyState.js";
```

# Processing Dashboard

<div class="fade-in">

Monitor real-time processing status, track progress, and analyze performance metrics.

</div>

## Processing Status

```js
// Get request ID from URL or input
const requestId = typeof Inputs !== "undefined" 
  ? await Inputs.text({label: "Request ID", value: ""})
  : new URLSearchParams(window.location.search).get("request_id") || "";

// Load status with error handling
let status = null;
let statusError = null;

if (requestId) {
  try {
    // Auto-refresh status every 2 seconds if processing
    async function* autoRefreshStatus(requestId) {
      if (!requestId) return null;
      
      while (true) {
        const statusData = await processingStatus(requestId);
        
        // Check if status is an error object
        if (statusData && statusData.error) {
          yield statusData;
          return;
        }
        
        yield statusData;
        
        // Stop refreshing if completed, failed, or no data
        if (!statusData || statusData.status === "completed" || statusData.status === "failed") {
          return statusData;
        }
        
        // Wait 2 seconds before next refresh
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    status = await autoRefreshStatus(requestId).next().then(r => r.value);
    
    // Check if status is an error object
    if (status && status.error) {
      statusError = status;
      status = null;
    }
  } catch (error) {
    statusError = { error: true, message: error.message || "Failed to load status" };
  }
}
```

```js
// Processing Status Card
html`<div class="card">
  ${statusError ? emptyStateError(statusError) : status ? html`
    <h3>Request: ${status.request_id}</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-top: 24px;">
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">
          ${status.statistics?.documents_processed || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Documents Processed</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: #34C759; margin-bottom: 4px;">
          ${status.statistics?.documents_succeeded || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Succeeded</div>
      </div>
      <div style="text-align: center;">
        <div style="font-size: 32px; font-weight: 600; color: ${status.statistics?.documents_failed > 0 ? '#FF3B30' : '#86868b'}; margin-bottom: 4px;">
          ${status.statistics?.documents_failed || 0}
        </div>
        <div style="font-size: 14px; color: #86868b;">Failed</div>
      </div>
    </div>
    <div style="margin-top: 24px;">
      <div style="font-size: 14px; color: #86868b; margin-bottom: 8px;">Progress</div>
      <div style="background: #f5f5f7; border-radius: 8px; height: 8px; overflow: hidden;">
        <div style="background: #007AFF; height: 100%; width: ${status.progress_percent || 0}%; transition: width 0.3s ease;"></div>
      </div>
      <div style="font-size: 14px; color: #86868b; margin-top: 8px; text-align: right;">
        ${status.progress_percent?.toFixed(1) || 0}%
      </div>
    </div>
  ` : emptyStateNoRequest()}
</div>`
```

## Progress Timeline

```js
// Progress Timeline Chart
// Design: Simple, beautiful, clear - one insight per chart
function progressTimelineChart(status) {
  if (!status || !status.completed_steps) return null;
  
  const steps = status.completed_steps.map((step, i) => ({
    step: step,
    index: i,
    timestamp: new Date(status.created_at).getTime() + (i * 1000), // Simulated
    completed: true
  }));
  
  return Plot.plot({
    // Generous whitespace
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    // Clean, minimal marks
    marks: [
      Plot.line(steps, {
        x: "timestamp",
        y: "index",
        stroke: "#007AFF", // Purposeful color (iOS blue)
        strokeWidth: 2,    // Subtle but clear
        curve: "natural"   // Smooth, organic feel
      }),
      Plot.dot(steps, {
        x: "timestamp",
        y: "index",
        fill: "#007AFF",
        r: 4
      })
    ],
    
    // Clear, readable labels
    x: {
      label: "Time",
      labelFontSize: 14,
      labelFontWeight: "500",
      tickFormat: (d) => new Date(d).toLocaleTimeString()
    },
    y: {
      label: "Step",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    
    // Generous sizing
    width: 800,
    height: 400,
    
    // Beautiful color scheme
    color: {
      scheme: "blues"
    }
  });
}

status ? progressTimelineChart(status) : null
```

## Step Completion

```js
// Step Completion Visualization
// Design: Clear progress, obvious completion
function stepCompletionChart(status) {
  if (!status) return null;
  
  const totalSteps = status.total_steps || 1;
  const completedSteps = status.completed_steps?.length || 0;
  const remainingSteps = totalSteps - completedSteps;
  
  const data = [
    {type: "Completed", value: completedSteps, color: "#34C759"},
    {type: "Remaining", value: remainingSteps, color: "#E5E5EA"}
  ];
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.barY(data, {
        x: "type",
        y: "value",
        fill: "color",
        rx: 4 // Rounded corners for beauty
      }),
      Plot.ruleY([0])
    ],
    
    x: {
      label: null,
      labelFontSize: 14
    },
    y: {
      label: "Steps",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    
    width: 400,
    height: 300,
    
    style: {
      background: "transparent"
    }
  });
}

status ? stepCompletionChart(status) : null
```

## Document Processing Timeline

```js
// Document Processing Over Time
// Design: Elegant area chart showing processing flow
function documentProcessingChart(status) {
  if (!status || !status.documents) return null;
  
  // Simulate processing timeline
  const timeline = status.documents.map((doc, i) => ({
    time: new Date(status.created_at).getTime() + (i * 5000),
    processed: i + 1,
    succeeded: doc.status === "success" ? i + 1 : (i > 0 ? timeline[i-1]?.succeeded || 0 : 0),
    failed: doc.status === "failed" ? (i > 0 ? (timeline[i-1]?.failed || 0) + 1 : 1) : (i > 0 ? timeline[i-1]?.failed || 0 : 0)
  }));
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.areaY(timeline, {
        x: "time",
        y: "processed",
        fill: "#007AFF",
        fillOpacity: 0.3,
        curve: "natural"
      }),
      Plot.line(timeline, {
        x: "time",
        y: "processed",
        stroke: "#007AFF",
        strokeWidth: 2
      }),
      Plot.areaY(timeline, {
        x: "time",
        y: "succeeded",
        fill: "#34C759",
        fillOpacity: 0.2,
        curve: "natural"
      }),
      Plot.line(timeline, {
        x: "time",
        y: "succeeded",
        stroke: "#34C759",
        strokeWidth: 2
      })
    ],
    
    x: {
      label: "Time",
      labelFontSize: 14,
      labelFontWeight: "500",
      tickFormat: (d) => new Date(d).toLocaleTimeString()
    },
    y: {
      label: "Documents",
      labelFontSize: 14,
      labelFontWeight: "500"
    },
    
    width: 800,
    height: 400
  });
}

status ? documentProcessingChart(status) : null
```

## Error Rate

```js
// Error Rate Visualization
// Design: Red only when needed, clear indication
function errorRateChart(status) {
  if (!status || !status.documents) return null;
  
  const errorRate = status.statistics?.documents_failed > 0
    ? (status.statistics.documents_failed / status.statistics.documents_processed) * 100
    : 0;
  
  const data = [
    {type: "Success Rate", value: 100 - errorRate, color: "#34C759"},
    {type: "Error Rate", value: errorRate, color: errorRate > 0 ? "#FF3B30" : "#E5E5EA"}
  ];
  
  return Plot.plot({
    margin: {top: 20, right: 20, bottom: 40, left: 60},
    
    marks: [
      Plot.barY(data, {
        x: "type",
        y: "value",
        fill: "color",
        rx: 4
      }),
      Plot.ruleY([0, 100])
    ],
    
    x: {
      label: null,
      labelFontSize: 14
    },
    y: {
      label: "Percentage",
      labelFontSize: 14,
      labelFontWeight: "500",
      domain: [0, 100],
      tickFormat: (d) => `${d}%`
    },
    
    width: 400,
    height: 300
  });
}

status ? errorRateChart(status) : null
```

## Current Step

```js
// Current Step Display
html`<div class="card">
  <h3>Current Status</h3>
  ${status ? html`
    <div style="margin-top: 16px;">
      <div style="font-size: 14px; color: #86868b; margin-bottom: 8px;">Current Step</div>
      <div style="font-size: 20px; font-weight: 600; color: #1d1d1f;">
        ${status.current_step || "Initializing..."}
      </div>
    </div>
    ${status.errors && status.errors.length > 0 ? html`
      <div style="margin-top: 24px; padding: 16px; background: #fff5f5; border-radius: 8px; border: 1px solid #ffebee;">
        <div style="font-size: 14px; font-weight: 600; color: #FF3B30; margin-bottom: 8px;">Errors</div>
        ${status.errors.map(error => html`
          <div style="font-size: 14px; color: #86868b; margin-bottom: 4px;">
            ${error.message || error}
          </div>
        `)}
      </div>
    ` : null}
  ` : html`
    <div class="empty-state">
      <p>No processing data available</p>
    </div>
  `}
</div>`
```

---

*Designed with simplicity, beauty, and intuition in mind.*

