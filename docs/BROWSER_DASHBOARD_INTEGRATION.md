# Browser Service & Dashboard Integration

## Current Architecture

### Two Separate Services

1. **Browser Service** (`services/browser/`)
   - **Purpose**: Electron-based browser shell for web automation
   - **Technology**: Electron + React + Vite
   - **Use Case**: Web scraping, navigation, automation, LocalAI chat
   - **UI**: React control panel with modules (Documents, Flows, LocalAI, Telemetry, Search)

2. **Perplexity Dashboard** (`services/orchestration/dashboard/`)
   - **Purpose**: Data visualization dashboard for Perplexity processing
   - **Technology**: Observable Framework + Plot + Runtime + Stdlib
   - **Use Case**: Visualizing processing results, analytics, knowledge graphs
   - **UI**: Static site with markdown-based pages

---

## Why They're Separate

### Different Purposes

**Browser Service**:
- Web automation and scraping
- Browser extension functionality
- Embedded control panel for gateway actions
- LocalAI chat interface
- Telemetry monitoring

**Perplexity Dashboard**:
- Data visualization and analytics
- Processing status monitoring
- Results exploration
- Knowledge graph visualization
- Query interface

### Different Technologies

**Browser Service**:
- Electron (desktop app)
- React (component framework)
- Vite (build tool)
- TypeScript

**Perplexity Dashboard**:
- Observable Framework (static site generator)
- Observable Plot (visualization library)
- Markdown-based pages
- JavaScript/ES modules

---

## Integration Options

### Option 1: Embed Dashboard in Browser Shell ✅ **RECOMMENDED**

Embed the Perplexity Dashboard as a module in the Browser Shell's React UI.

**Benefits**:
- Single unified interface
- Access to both automation and visualization
- Consistent user experience
- Share authentication/state

**Implementation**:
```typescript
// services/browser/shell/ui/src/modules/Perplexity/
// Add new module to Browser Shell

import { useEffect, useRef } from 'react';

export function PerplexityModule() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  
  return (
    <iframe
      ref={iframeRef}
      src="http://localhost:3000" // Observable Framework dashboard
      style={{ width: '100%', height: '100%', border: 'none' }}
      title="Perplexity Dashboard"
    />
  );
}
```

**Steps**:
1. Add Perplexity module to Browser Shell
2. Embed Observable Framework dashboard via iframe
3. Add navigation link in Browser Shell sidebar
4. Share authentication/API keys between services

---

### Option 2: Embed Browser Shell in Dashboard

Embed the Browser Shell's automation capabilities in the Perplexity Dashboard.

**Benefits**:
- Visualization-first experience
- Access to automation from dashboard
- Unified data exploration

**Implementation**:
```javascript
// services/orchestration/dashboard/src/browser.md
// Add browser automation page to dashboard

import {html} from "@observablehq/stdlib";

// Embed browser automation iframe
html`<iframe 
  src="http://localhost:4173" 
  style="width: 100%; height: 800px; border: none;"
></iframe>`
```

---

### Option 3: Unified Gateway Integration

Both services connect to the same gateway/API, providing unified access.

**Benefits**:
- Shared API endpoints
- Consistent data access
- Single source of truth

**Current State**:
- Browser Shell: Connects to gateway (`SHELL_GATEWAY_URL`)
- Perplexity Dashboard: Connects to Perplexity API (`PERPLEXITY_API_BASE`)

**Enhancement**:
- Both use same gateway base URL
- Shared authentication
- Unified API documentation

---

### Option 4: Browser Shell as Dashboard Host

Use Browser Shell as the primary interface, with dashboard embedded.

**Benefits**:
- Single application
- Rich automation capabilities
- Embedded visualization

**Implementation**:
- Add Perplexity Dashboard as Browser Shell module
- Use Electron's BrowserView for dashboard
- Share state between modules

---

## Recommended Approach

### **Option 1: Embed Dashboard in Browser Shell** ✅

**Why**:
1. Browser Shell is the primary user interface
2. Dashboard adds visualization capabilities
3. Unified experience
4. Easy to implement

**Implementation Plan**:

1. **Add Perplexity Module to Browser Shell**:
   ```typescript
   // services/browser/shell/ui/src/modules/Perplexity/PerplexityModule.tsx
   ```

2. **Embed Observable Framework Dashboard**:
   - Iframe or direct integration
   - Share API endpoints
   - Unified navigation

3. **Add Navigation Link**:
   - Add "Perplexity Dashboard" to Browser Shell sidebar
   - Link to embedded dashboard module

4. **Share Configuration**:
   - Use same API base URL
   - Share authentication tokens
   - Unified settings

---

## Quick Integration Steps

### Step 1: Add Perplexity Module to Browser Shell

```bash
cd services/browser/shell/ui/src/modules
mkdir Perplexity
```

Create `PerplexityModule.tsx`:
```typescript
import { useEffect, useRef } from 'react';
import styles from './PerplexityModule.module.css';

export function PerplexityModule() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const dashboardUrl = import.meta.env.VITE_DASHBOARD_URL || 'http://localhost:3000';
  
  return (
    <div className={styles.container}>
      <iframe
        ref={iframeRef}
        src={dashboardUrl}
        className={styles.iframe}
        title="Perplexity Dashboard"
        allow="fullscreen"
      />
    </div>
  );
}
```

### Step 2: Add to Browser Shell Navigation

Update `services/browser/shell/ui/src/components/NavPanel.tsx`:
```typescript
// Add Perplexity to navigation
{ name: 'Perplexity', icon: '📊', module: 'Perplexity' }
```

### Step 3: Register Module

Update `services/browser/shell/ui/src/App.tsx`:
```typescript
import { PerplexityModule } from './modules/Perplexity/PerplexityModule';

// Add to module routing
case 'Perplexity':
  return <PerplexityModule />;
```

### Step 4: Share API Configuration

Update Browser Shell to use same API base:
```typescript
// Use same API base URL
const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8080';
```

---

## Benefits of Integration

### For Users
- ✅ Single interface for all tools
- ✅ Seamless navigation
- ✅ Consistent experience
- ✅ No context switching

### For Development
- ✅ Shared authentication
- ✅ Unified API access
- ✅ Consistent design system
- ✅ Easier maintenance

### For Architecture
- ✅ Modular design
- ✅ Reusable components
- ✅ Clear separation of concerns
- ✅ Easy to extend

---

## Current State

**Browser Service**:
- ✅ Electron shell running
- ✅ React UI with modules
- ✅ Gateway integration
- ✅ LocalAI chat

**Perplexity Dashboard**:
- ✅ Observable Framework running
- ✅ Three dashboards (Processing, Results, Analytics)
- ✅ Design system in place
- ✅ API integration ready

**Integration Status**: ⏳ **Not Yet Integrated**

---

## Next Steps

1. **Decide on Integration Approach**
   - Option 1: Embed Dashboard in Browser Shell (Recommended)
   - Option 2: Embed Browser Shell in Dashboard
   - Option 3: Keep separate, unified gateway

2. **Implement Integration**
   - Add Perplexity module to Browser Shell
   - Configure shared API endpoints
   - Test unified experience

3. **Enhance Integration**
   - Share authentication
   - Unified navigation
   - Consistent design system

---

## Questions to Consider

1. **Primary Interface**: Which should be the main interface?
   - Browser Shell (automation-focused)
   - Dashboard (visualization-focused)
   - Both (modular approach)

2. **User Workflow**: How do users typically work?
   - Automation first, then visualization
   - Visualization first, then automation
   - Both simultaneously

3. **Deployment**: How will they be deployed?
   - Together (single app)
   - Separate (different ports)
   - Embedded (one in the other)

---

**Recommendation**: **Option 1 - Embed Dashboard in Browser Shell**

This provides the best user experience with a unified interface while maintaining the strengths of both services.

