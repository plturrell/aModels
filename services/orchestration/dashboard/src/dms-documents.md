---
title: DMS Documents Library
description: Browse and explore uploaded documents
---

```js
import * as Plot from "@observablehq/plot";
import dmsDocumentsData from "../data/loaders/dms-documents.js";
import {html} from "@observablehq/stdlib";
import {emptyStateNoData} from "../components/emptyState.js";
```

# DMS Documents Library

<div class="fade-in">

Browse your document library, view processing status, and access document details.

</div>

## Documents List

```js
// Load documents
const documentsData = await dmsDocumentsData({limit: 50, offset: 0});
```

```js
html`<div class="card">
  <h3>Documents (${documentsData.total || 0})</h3>
  ${documentsData.error ? html`
    <div style="padding: 24px; text-align: center; color: #FF3B30;">
      Error: ${documentsData.message}
    </div>
  ` : documentsData.documents && documentsData.documents.length > 0 ? html`
    <div style="margin-top: 16px;">
      ${documentsData.documents.map(doc => html`
        <div style="padding: 16px; background: #F2F2F7; border-radius: 8px; margin-bottom: 12px; cursor: pointer; transition: background 0.2s;" 
             onmouseover="this.style.background='#E5E5EA'" 
             onmouseout="this.style.background='#F2F2F7'">
          <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
            <div style="flex: 1;">
              <div style="font-size: 18px; font-weight: 600; color: #1d1d1f; margin-bottom: 4px;">
                ${doc.name}
              </div>
              <div style="font-size: 14px; color: #86868b;">
                ${doc.description || 'No description'}
              </div>
            </div>
            <div style="padding: 4px 12px; background: ${doc.catalog_identifier ? '#34C759' : '#FF9500'}; color: white; border-radius: 12px; font-size: 12px; font-weight: 500;">
              ${doc.catalog_identifier ? 'Processed' : 'Pending'}
            </div>
          </div>
          <div style="font-size: 12px; color: #86868b; margin-top: 8px;">
            Created: ${new Date(doc.created_at).toLocaleString()} •
            ID: ${doc.id}
          </div>
        </div>
      `)}
    </div>
  ` : emptyStateNoData("No documents found")}
</div>`
```

## Document Status Distribution

```js
// Status distribution
const statusData = documentsData.documents ? [
  {status: "Processed", count: documentsData.documents.filter(d => d.catalog_identifier).length, color: "#34C759"},
  {status: "Pending", count: documentsData.documents.filter(d => !d.catalog_identifier).length, color: "#FF9500"}
].filter(d => d.count > 0) : [];

Plot.plot({
  title: "Document Status Distribution",
  width: 400,
  height: 300,
  marginLeft: 100,
  y: {label: "Status"},
  x: {label: "Count"},
  color: {domain: statusData.map(d => d.status), range: statusData.map(d => d.color)},
  marks: [
    Plot.barX(statusData, {
      x: "count",
      y: "status",
      fill: "color"
    })
  ]
})
```

## Documents Over Time

```js
// Documents created over time
const timeData = documentsData.documents ? 
  documentsData.documents.map(doc => ({
    date: new Date(doc.created_at),
    count: 1
  })).sort((a, b) => a.date - b.date) : [];

Plot.plot({
  title: "Documents Uploaded Over Time",
  width: 800,
  height: 300,
  marginLeft: 60,
  x: {label: "Date", type: "time"},
  y: {label: "Count"},
  marks: [
    Plot.lineY(timeData, {
      x: "date",
      y: "count",
      stroke: "#007AFF",
      strokeWidth: 2
    }),
    Plot.dot(timeData, {
      x: "date",
      y: "count",
      fill: "#007AFF",
      r: 4
    })
  ]
})
```

