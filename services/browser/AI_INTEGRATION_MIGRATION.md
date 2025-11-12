# AI Integration Migration Guide

## Overview

This guide outlines the migration from scattered AI service calls to a unified, type-safe AI integration layer.

## Current Issues

1. **Scattered API calls** across multiple files
2. **No caching** or intelligent routing
3. **Type safety** issues
4. **Error handling** inconsistencies
5. **Service discovery** problems

## New Architecture

### **Unified AI Service Layer**
- **Single entry point** for all AI operations
- **Type-safe interfaces** with TypeScript
- **Intelligent caching** with TTL
- **Automatic retry** and fallback
- **Service health monitoring**

### **Service Registry**
```typescript
// Before: Scattered URLs
const gnnUrl = 'http://localhost:8080/gnn/...'
const gooseUrl = 'http://localhost:8081/goose/...'

// After: Unified service
const response = await aiService.query({
  type: 'hybrid',
  query: 'Analyze compliance risks',
  context: { graphData, regulatoryScope }
});
```

## Migration Steps

### **1. Replace Direct API Calls**

**Before** (in components):
```typescript
// Old way - scattered
const gnnResponse = await fetch('/gnn/embeddings', {...});
const gooseResponse = await fetch('/goose/execute', {...});
```

**After** (using hooks):
```typescript
// New way - unified
const { data, loading, error } = useGNNAnalysis();
const result = await analyzeGraph(graphData);
```

### **2. Update Components**

**Before**:
```typescript
const [data, setData] = useState(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);

// Manual fetching
```

**After**:
```typescript
const { data, loading, error, analyzeGraph } = useGNNAnalysis();
```

### **3. Service Registration**

Add the new AI handler to the browser shell backend:

```go
// In main.go
aiHandler := NewAIHandler()
aiHandler.RegisterRoutes(router)
```

## Usage Examples

### **GNN Analysis**
```typescript
const { data, loading, error, analyzeGraph } = useGNNAnalysis();

const handleAnalysis = async () => {
  const result = await analyzeGraph({
    graph: graphData,
    task: 'structural-insights'
  });
  
  console.log('GNN insights:', result);
};
```

### **Compliance Audit**
```typescript
const { data, loading, error, executeTask } = useGooseTask();

const runAudit = async () => {
  const audit = await executeTask({
    task: 'BCBS239 compliance audit',
    context: { principles: ['P3', 'P4', 'P7', 'P12'] },
    autoRemediate: true
  });
};
```

### **Deep Research**
```typescript
const { data, loading, error, research } = useDeepResearch();

const analyzeRegulations = async () => {
  const findings = await research({
    query: 'Latest Basel III requirements',
    scope: 'regulatory',
    sources: ['basel-committee', 'fsi-papers']
  });
};
```

### **Hybrid Query**
```typescript
const { data, loading, error, query } = useHybridAI();

const comprehensiveAnalysis = async () => {
  const result = await query(
    'Analyze this graph for compliance risks and generate remediation scripts',
    { graphData, complianceScope: 'BCBS239' }
  );
};
```

## Benefits

### **1. Developer Experience**
- **Type safety** throughout
- **Consistent error handling**
- **Built-in caching**
- **Easy testing**

### **2. Performance**
- **Response caching** (5min TTL)
- **Automatic retries**
- **Request deduplication**
- **Progressive enhancement**

### **3. Maintainability**
- **Single source of truth**
- **Easy service switching**
- **Centralized configuration**
- **Health monitoring**

### **4. Scalability**
- **Service discovery**
- **Load balancing**
- **Circuit breakers**
- **Rate limiting**

## Testing

### **Unit Tests**
```typescript
import { aiService } from '../services/AIIntegration';

describe('AI Service', () => {
  it('should handle GNN queries', async () => {
    const result = await aiService.gnnAnalysis({
      graph: mockGraph,
      task: 'embeddings'
    });
    expect(result.data).toBeDefined();
  });
});
```

### **Integration Tests**
```typescript
import { renderHook, act } from '@testing-library/react-hooks';
import { useGNNAnalysis } from '../hooks/useAI';

describe('useGNNAnalysis', () => {
  it('should handle loading states', async () => {
    const { result } = renderHook(() => useGNNAnalysis());
    
    act(() => {
      result.current.analyzeGraph(mockData);
    });
    
    expect(result.current.loading).toBe(true);
  });
});
```

## Rollback Plan

1. **Keep old APIs** during transition
2. **Feature flags** for gradual rollout
3. **Monitoring** for performance regression
4. **Easy rollback** via configuration

## Timeline

- **Phase 1**: Implement new service layer (1-2 days)
- **Phase 2**: Migrate critical components (2-3 days)
- **Phase 3**: Full migration and testing (1 week)
- **Phase 4**: Remove old APIs (1 day)

## Migration Checklist

- [ ] Implement AIIntegration service
- [ ] Create React hooks
- [ ] Add backend orchestrator
- [ ] Update components to use new hooks
- [ ] Add comprehensive tests
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Team training
- [ ] Production deployment
- [ ] Monitor and optimize
