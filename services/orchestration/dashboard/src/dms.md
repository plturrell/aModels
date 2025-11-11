---
title: DMS Dashboard
description: Document Management System - Upload, process, and explore documents
---

```js
import {html} from "@observablehq/stdlib";
```

# DMS Dashboard

<div class="fade-in">

Welcome to the Document Management System dashboard. Upload documents and track their processing through the full pipeline.

</div>

## Quick Navigation

```js
html`<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; margin-top: 32px;">
  <a href="/dms-processing" class="card-link">
    <div class="card">
      <div style="font-size: 32px; margin-bottom: 8px;">📊</div>
      <h3>Processing</h3>
      <p>Real-time processing status and progress</p>
    </div>
  </a>
  
  <a href="/dms-results" class="card-link">
    <div class="card">
      <div style="font-size: 32px; margin-bottom: 8px;">📄</div>
      <h3>Results</h3>
      <p>Explore processed documents and intelligence</p>
    </div>
  </a>
  
  <a href="/dms-analytics" class="card-link">
    <div class="card">
      <div style="font-size: 32px; margin-bottom: 8px;">📈</div>
      <h3>Analytics</h3>
      <p>Analyze trends and patterns</p>
    </div>
  </a>
  
  <a href="/dms-documents" class="card-link">
    <div class="card">
      <div style="font-size: 32px; margin-bottom: 8px;">📚</div>
      <h3>Documents</h3>
      <p>Browse document library</p>
    </div>
  </a>
</div>`
```

## Design Philosophy

This dashboard follows the **Jobs & Ive lens**:
- **Simplicity**: Clean, focused interfaces
- **Beauty**: Elegant typography, generous whitespace
- **Intuition**: Zero learning curve
- **Delight**: Smooth animations, beautiful interactions

---

## Getting Started

1. **Upload Documents**: Use the DMS API or Browser Shell to upload documents
2. **Track Processing**: Monitor real-time processing status
3. **Explore Results**: View intelligence, relationships, and patterns
4. **Analyze Trends**: Understand document processing performance

---

## Features

- ✅ Real-time processing status
- ✅ Intelligence visualization
- ✅ Relationship networks
- ✅ Pattern discovery
- ✅ Export functionality
- ✅ Deep linking support

