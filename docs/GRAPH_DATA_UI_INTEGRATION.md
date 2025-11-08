# Graph Data UI Integration & User Experience

## Overview

The unified `GraphData` format (`services/graph/pkg/models/graph_data.go` and `services/training/api/graph_data_models.py`) provides a standardized data structure that flows seamlessly from backend services through API endpoints to frontend UI components, creating a consistent user experience across all graph-related features.

## Data Flow Architecture

```
Backend Services → API Endpoints → Frontend Components → User Interface
     ↓                  ↓                ↓                    ↓
GraphData         JSON Response    React Components    Visualizations
(Go/Python)       (Unified)        (TypeScript)        (Charts/Graphs)
```

## User Journey Integration

### 1. **Knowledge Graph Processing Journey**

**Backend Flow:**
- User uploads documents or triggers knowledge graph processing
- `ProcessKnowledgeGraphNode` (Go) processes data and returns unified `GraphData`
- Data is stored in state as both `graph_data` (unified) and `knowledge_graph` (legacy compatibility)

**API Endpoint:**
- `POST /knowledge-graph/process` (services/graph/cmd/graph-server/main.go:190)
- Returns `GraphData` in JSON format

**Frontend Integration:**
- Dashboard loader: `services/orchestration/dashboard/data/loaders/graph.js`
- Fetches from `/api/perplexity/graph/{requestId}/relationships`
- Converts to visualization format

**User Experience:**
1. User initiates document processing
2. System processes and builds knowledge graph
3. User sees graph statistics (nodes, edges, relationships)
4. Interactive visualization shows relationships
5. User can explore node types, relationship types, confidence scores

**UI Components:**
- **Knowledge Graph Dashboard (`services/orchestration/dashboard/src/graph.md`)**
  - Displays graph statistics (nodes, edges, relationships, avg confidence)
  - Shows node type distribution charts
  - Shows relationship type distribution
  - Lists relationships with confidence scores
  - Uses Observable Plot for visualizations

### 2. **Search & Discovery Journey**

**Backend Flow:**
- User performs unified search query
- Search service queries knowledge graph using `GraphData` format
- Results include graph context and relationships

**API Endpoint:**
- `POST /search/unified` (includes `knowledge_graph` in sources)
- Returns search results with graph metadata

**Frontend Integration:**
- `services/browser/shell/ui/src/modules/Search/SearchModule.tsx`
- `services/browser/shell/ui/src/api/search.ts`
- Includes `knowledge_graph` as a search source

**User Experience:**
1. User enters search query
2. System searches across multiple sources including knowledge graph
3. Results show graph-related entities and relationships
4. User can explore connections between search results
5. Graph insights enhance search relevance

**UI Components:**
- **Search Module** displays unified results
- Graph entities highlighted in search results
- Relationship indicators show connections

### 3. **Document Processing Journey (DMS/Perplexity/Murex)**

**Backend Flow:**
- Documents processed through DMS, Perplexity, or Murex pipelines
- Knowledge graph extracted and stored as `GraphData`
- Intelligence metadata includes graph statistics

**API Endpoints:**
- Various endpoints return intelligence with `knowledge_graph_nodes` and `knowledge_graph_edges`
- Format: `{ knowledge_graph_nodes: number, knowledge_graph_edges: number }`

**Frontend Integration:**
- `services/browser/shell/ui/src/modules/DMS/views/ResultsView.tsx`
- `services/browser/shell/ui/src/modules/Perplexity/views/ResultsView.tsx`
- `services/browser/shell/ui/src/modules/Murex/views/ResultsView.tsx`
- All display graph statistics in intelligence panels

**User Experience:**
1. User uploads/processes documents
2. System extracts knowledge graph
3. Results view shows:
   - Number of nodes discovered
   - Number of edges/relationships found
   - Graph quality indicators
4. User can drill into graph details
5. Graph insights inform document understanding

**UI Components:**
- **Results View** shows intelligence metrics including:
  - `knowledge_graph_nodes` count
  - `knowledge_graph_edges` count
  - Graph quality score (if available)

### 4. **GNN Analysis Journey**

**Backend Flow:**
- User queries GNN for structural insights
- `QueryGNNNode` accepts `GraphData` from state
- GNN service processes unified format
- Results converted back to `GraphData` format

**API Endpoints:**
- `POST /gnn/query` (services/graph/cmd/graph-server/main.go:332)
- `POST /gnn/hybrid-query` (services/graph/cmd/graph-server/main.go:368)
- Both accept and return `GraphData` format

**Frontend Integration:**
- Currently backend-only, but ready for frontend integration
- Future: GNN insights visualization components

**User Experience (Planned):**
1. User requests GNN analysis on graph
2. System performs structural analysis
3. User sees:
   - Anomaly detections
   - Structural patterns
   - Link predictions
   - Node classifications
4. Insights enhance graph understanding

### 5. **Relational Data Analysis Journey**

**Backend Flow:**
- Relational data processed and converted to graph
- `GraphData` format ensures consistency
- Statistics tracked in intelligence metadata

**Frontend Integration:**
- `services/browser/shell/ui/src/modules/Relational/views/ResultsView.tsx`
- Displays graph statistics from relational processing

**User Experience:**
1. User processes relational data (tables, schemas)
2. System builds knowledge graph representation
3. User sees graph statistics
4. Can explore entity relationships
5. Graph helps understand data lineage

## UI Component Mapping

### Current Components Using Graph Data

1. **Knowledge Graph Dashboard** (`services/orchestration/dashboard/src/graph.md`)
   - **Data Source:** `loadGraph()` function
   - **Format:** Converts relationships to nodes/edges
   - **Visualization:** Observable Plot charts, relationship cards
   - **User Actions:** Explore relationships, view statistics

2. **Search Module** (`services/browser/shell/ui/src/modules/Search/SearchModule.tsx`)
   - **Data Source:** Unified search API with `knowledge_graph` source
   - **Format:** Search results with graph context
   - **Visualization:** Search result cards with relationship indicators
   - **User Actions:** Search, filter, explore connections

3. **Results Views** (DMS/Perplexity/Murex/Relational)
   - **Data Source:** Intelligence metadata with graph statistics
   - **Format:** `{ knowledge_graph_nodes: number, knowledge_graph_edges: number }`
   - **Visualization:** Statistics badges, metrics panels
   - **User Actions:** View processing results, drill into details

4. **SGMI Flow Visualization** (`services/browser/shell/ui/src/App.jsx`)
   - **Data Source:** SGMI flow data (nodes/links format)
   - **Format:** Custom format, but compatible with `GraphData` structure
   - **Visualization:** Sankey diagrams, network graphs (nivo/react)
   - **User Actions:** Click nodes/edges, explore transitions

### Future UI Components (Ready for Integration)

1. **GNN Insights Panel**
   - **Data Source:** `/gnn/query` and `/gnn/hybrid-query` endpoints
   - **Format:** `GraphData` with GNN analysis results
   - **Visualization:** Anomaly highlights, pattern indicators, prediction scores
   - **User Actions:** View insights, filter by confidence, explore predictions

2. **Graph Explorer**
   - **Data Source:** `/knowledge-graph/query` endpoint
   - **Format:** `GraphData` with nodes, edges, metadata
   - **Visualization:** Interactive graph visualization (D3.js, Cytoscape, or similar)
   - **User Actions:** Navigate graph, filter by type, search nodes

3. **Quality Dashboard**
   - **Data Source:** `GraphData.quality` field
   - **Format:** Quality metrics (score, level, issues, recommendations)
   - **Visualization:** Quality score indicators, issue lists, recommendations
   - **User Actions:** Review quality, address issues, apply recommendations

## Benefits of Unified Format for UX

### 1. **Consistency**
- **Before:** Different formats for different services (Neo4j format, GNN format, custom formats)
- **After:** Single `GraphData` format across all services
- **UX Impact:** Users see consistent data structure, reducing confusion

### 2. **Reliability**
- **Before:** Format mismatches caused display errors**
- **After:** Validation ensures data integrity before reaching UI
- **UX Impact:** Fewer errors, more reliable visualizations

### 3. **Performance**
- **Before:** Multiple conversion steps, potential data loss
- **After:** Direct conversion, optimized for UI consumption
- **UX Impact:** Faster load times, smoother interactions

### 4. **Extensibility**
- **Before:** Adding new graph features required format changes in multiple places
- **After:** Extend `GraphData` model, UI automatically benefits
- **UX Impact:** New features appear consistently across UI

### 5. **Debugging**
- **Before:** Format issues hard to trace across services
- **After:** Single source of truth, clear validation errors
- **UX Impact:** Better error messages, easier troubleshooting

## Data Format Examples

### Backend → Frontend Flow

**Go Backend Response:**
```json
{
  "graph_data": {
    "nodes": [
      {
        "id": "node_1",
        "type": "table",
        "label": "Customers",
        "properties": { "schema": "public", "row_count": 1000 }
      }
    ],
    "edges": [
      {
        "source": "node_1",
        "target": "node_2",
        "label": "references",
        "properties": { "foreign_key": "customer_id" }
      }
    ],
    "metadata": {
      "project_id": "project_123",
      "system_id": "system_456",
      "root_node_id": "node_1"
    },
    "quality": {
      "score": 0.85,
      "level": "good",
      "issues": [],
      "recommendations": ["Add more relationships"]
    }
  }
}
```

**Frontend Consumption:**
```typescript
// TypeScript interface matching GraphData
interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Metadata;
  quality?: Quality;
}

// Component usage
const { data } = useApiData<GraphData>('/knowledge-graph/process');
const nodeCount = data?.nodes.length || 0;
const edgeCount = data?.edges.length || 0;
```

## Integration Points

### 1. **API Endpoints Serving GraphData**

| Endpoint | Method | Returns GraphData | UI Component |
|----------|--------|-------------------|--------------|
| `/knowledge-graph/process` | POST | ✅ Yes | Dashboard |
| `/gnn/query` | POST | ✅ Yes | GNN Insights (future) |
| `/gnn/hybrid-query` | POST | ✅ Yes | Hybrid Analysis (future) |
| `/api/perplexity/graph/{id}/relationships` | GET | ⚠️ Legacy format | Dashboard |
| `/search/unified` | POST | ⚠️ Metadata only | Search Module |

### 2. **Frontend Data Loaders**

- `services/orchestration/dashboard/data/loaders/graph.js`
  - Loads graph data for dashboard
  - **TODO:** Update to use unified `GraphData` format

### 3. **React Components**

- Search Module: Uses graph metadata in search results
- Results Views: Display graph statistics
- Dashboard: Full graph visualization (needs format update)

## Migration Path for UI

### Phase 1: Backend Ready ✅
- Unified `GraphData` format implemented
- API endpoints return unified format
- Backward compatibility maintained

### Phase 2: Frontend Updates (Recommended)
1. **Update Dashboard Loader**
   - Modify `loadGraph()` to handle `GraphData` format
   - Update visualization components to use unified structure

2. **Create GraphData TypeScript Types**
   - Add types matching Go/Python models
   - Ensure type safety across frontend

3. **Update Results Views**
   - Use `GraphData` instead of legacy format
   - Display quality metrics from `GraphData.quality`

4. **Add GNN Insights UI**
   - Create components for GNN analysis results
   - Visualize anomalies, patterns, predictions

### Phase 3: Enhanced Features
1. **Graph Explorer Component**
   - Interactive graph visualization
   - Filter, search, navigate nodes/edges
   - Show metadata and quality indicators

2. **Quality Dashboard**
   - Visualize quality metrics
   - Show issues and recommendations
   - Track quality over time

## User Experience Improvements

### Immediate Benefits
1. **Consistent Data Structure:** Users see same format everywhere
2. **Better Error Handling:** Validation prevents bad data from reaching UI
3. **Performance:** Optimized conversion reduces latency

### Future Benefits
1. **Rich Metadata:** Quality scores, recommendations visible to users
2. **Enhanced Visualizations:** Consistent format enables better graph rendering
3. **Cross-Service Integration:** Easier to combine data from multiple sources

## Recommendations

### Short Term
1. Update dashboard loader to use `GraphData` format
2. Add TypeScript types for `GraphData`
3. Update results views to display quality metrics

### Medium Term
1. Create Graph Explorer component
2. Add GNN insights visualization
3. Implement quality dashboard

### Long Term
1. Real-time graph updates via WebSocket
2. Collaborative graph editing
3. Advanced graph analytics UI

## Conclusion

The unified `GraphData` format provides a solid foundation for consistent, reliable graph data presentation across all UI components. While the backend is fully integrated, the frontend components are ready to leverage this unified format for improved user experience, better performance, and enhanced features.

The format's design ensures:
- **Backward Compatibility:** Legacy formats still work
- **Forward Compatibility:** Easy to extend with new features
- **Type Safety:** Consistent structure across Go, Python, and TypeScript
- **User Experience:** Consistent, reliable, performant graph interactions

