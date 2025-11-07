---
title: Murex Processing Dashboard
description: Real-time processing status, progress, and statistics
---

```js
import * as Plot from "@observablehq/plot";
import murexProcessingStatus from "../data/loaders/murex-processing.js";
import {html} from "@observablehq/stdlib";
import {emptyStateNoRequest, emptyStateLoading, emptyStateError} from "../components/emptyState.js";
```

# Murex Processing Dashboard

<div class="fade-in">

Monitor Murex trade processing in real-time. Track progress, view statistics, and identify issues.

</div>

## Processing Status

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

// Load status with error handling
let status = null;
let statusError = null;

if (requestId) {
  try {
    // Auto-refresh status every 2 seconds if processing
    async function* autoRefreshStatus(requestId) {
      if (!requestId) return null;
      
      while (true) {
        const statusData = await murexProcessingStatus(requestId);
        
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
        <div style="font-size: 14px; color: #86868b;">Trades Processed</div>
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
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div style="font-size: 14px; color: #86868b;">Progress</div>
        <div style="font-size: 14px; color: #86868b; text-align: right;">
          ${status.progress_percent?.toFixed(1) || 0}%
        </div>
      </div>
      <div style="height: 8px; background: #F2F2F7; border-radius: 4px; overflow: hidden;">
        <div style="height: 100%; width: ${status.progress_percent || 0}%; background: #007AFF; transition: width 0.3s ease;"></div>
      </div>
    </div>
    <div style="margin-top: 24px; padding: 16px; background: #F2F2F7; border-radius: 8px;">
      <div style="font-size: 14px; color: #86868b; margin-bottom: 4px;">Current Step</div>
      <div style="font-size: 20px; font-weight: 600; color: #1d1d1f;">
        ${status.current_step || "Initializing..."}
      </div>
    </div>
  ` : emptyStateNoRequest()}
</div>`
```

## Progress Timeline

```js
// Progress timeline chart
const progressData = status && status.completed_steps ? status.completed_steps.map((step, i) => ({
  step: step,
  index: i + 1,
  timestamp: new Date(status.created_at).getTime() + (i * 1000)
})) : [];

Plot.plot({
  title: "Processing Steps Timeline",
  width: 800,
  height: 300,
  marginLeft: 80,
  x: {label: "Time", type: "linear"},
  y: {label: "Step", domain: progressData.map(d => d.step)},
  marks: [
    Plot.line(progressData, {
      x: "timestamp",
      y: "step",
      stroke: "#007AFF",
      strokeWidth: 2
    }),
    Plot.dot(progressData, {
      x: "timestamp",
      y: "step",
      fill: "#007AFF",
      r: 4
    })
  ]
})
```

## Trade Processing Status

```js
// Trade status distribution
const tradeStatusData = status && status.documents ? [
  {status: "Succeeded", count: status.statistics?.documents_succeeded || 0, color: "#34C759"},
  {status: "Failed", count: status.statistics?.documents_failed || 0, color: "#FF3B30"},
  {status: "Processing", count: (status.statistics?.documents_processed || 0) - (status.statistics?.documents_succeeded || 0) - (status.statistics?.documents_failed || 0), color: "#007AFF"}
].filter(d => d.count > 0) : [];

Plot.plot({
  title: "Trade Status Distribution",
  width: 400,
  height: 300,
  marginLeft: 100,
  y: {label: "Status"},
  x: {label: "Count"},
  color: {domain: tradeStatusData.map(d => d.status), range: tradeStatusData.map(d => d.color)},
  marks: [
    Plot.barX(tradeStatusData, {
      x: "count",
      y: "status",
      fill: "color"
    })
  ]
})
```

## Errors

```js
html`<div class="card">
  ${status && status.errors && status.errors.length > 0 ? html`
    <h3>Errors</h3>
    <div style="margin-top: 16px;">
      ${status.errors.map((error, i) => html`
        <div style="padding: 12px; background: #fff5f5; border-radius: 8px; border: 1px solid #ffebee; margin-bottom: 8px;">
          <div style="font-size: 14px; font-weight: 600; color: #FF3B30; margin-bottom: 4px;">
            ${typeof error === 'object' && error.code ? error.code : 'Error'}
          </div>
          <div style="font-size: 14px; color: #86868b;">
            ${typeof error === 'object' && error.message ? error.message : error}
          </div>
        </div>
      `)}
    </div>
  ` : html`
    <div style="text-align: center; padding: 32px; color: #86868b;">
      No errors
    </div>
  `}
</div>`
```

