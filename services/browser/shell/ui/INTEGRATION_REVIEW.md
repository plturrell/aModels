# Backend to UI Integration Review

## Document Management Service (DMS) Integration

### Endpoint Configuration
- **UI Endpoint**: `/dms/documents`
- **Shell Proxy**: Routes `/dms/*` to DMS service (strips `/dms` prefix)
- **Backend Endpoint**: `/documents/` (GET)
- **Backend Service**: FastAPI DMS service on port 8096 (host) / 8080 (container)

### Data Flow
1. UI calls `/dms/documents` (relative path, uses `API_BASE` from env)
2. Shell server proxies to `SHELL_DMS_ENDPOINT/documents`
3. DMS FastAPI returns `List[DocumentRead]`
4. FastAPI automatically serializes `datetime` fields to ISO 8601 strings
5. UI receives `DocumentRecord[]` with string timestamps

### Schema Mapping
**Backend (`DocumentRead`)**:
```python
class DocumentRead(BaseModel):
    id: str
    name: str
    description: Optional[str]
    storage_path: str
    catalog_identifier: Optional[str]
    extraction_summary: Optional[str]
    created_at: datetime  # Serialized to ISO string
    updated_at: datetime  # Serialized to ISO string
```

**Frontend (`DocumentRecord`)**:
```typescript
interface DocumentRecord {
  id: string;
  name: string;
  description?: string | null;
  storage_path: string;
  catalog_identifier?: string | null;
  extraction_summary?: string | null;
  created_at: string;  // ISO 8601 datetime string
  updated_at: string;  // ISO 8601 datetime string
}
```

### Improvements Made

1. **Enhanced Error Handling** (`api/client.ts`):
   - Better error message parsing (handles FastAPI error format)
   - Network error detection with helpful messages
   - Empty response handling
   - Structured error extraction from JSON responses

2. **Improved UI Error Display** (`DocumentsModule.tsx`):
   - More detailed error messages
   - Retry button for failed requests
   - Better error formatting

3. **API Documentation**:
   - Added docstring to `list_documents` endpoint

### Testing Checklist

- [x] Endpoint routing verified (`/dms/documents` → `/documents`)
- [x] Datetime serialization confirmed (FastAPI handles automatically)
- [x] Error handling improved
- [x] Type definitions match backend schema
- [ ] Test with real DMS service (requires running service)
- [ ] Test error scenarios (service down, network errors)
- [ ] Test empty response handling

### Configuration

**Shell Server Environment Variables**:
- `SHELL_DMS_ENDPOINT` - DMS service URL (defaults to `${SHELL_GATEWAY_URL}/dms`)
- `SHELL_GATEWAY_URL` - Gateway base URL (defaults to `http://localhost:8000`)

**UI Environment Variables**:
- `VITE_SHELL_API` - API base URL (defaults to empty string for relative paths)

## LocalAI Integration

### Endpoint Configuration
- **UI Chat Endpoint**: `/localai/v1/chat/completions`
- **UI Models Endpoint**: `/api/localai/models`
- **Shell Proxy**: Routes `/localai/*` to LocalAI service (strips `/localai` prefix)
- **Backend Service**: LocalAI service (OpenAI-compatible API)

### Data Flow
1. UI calls `/localai/v1/chat/completions` (relative path)
2. Shell server proxies to `SHELL_LOCALAI_URL/v1/chat/completions`
3. LocalAI returns OpenAI-compatible `ChatResponse`
4. UI processes response and extracts citations/followups

### Schema Mapping
**Backend (OpenAI-compatible)**:
- Uses OpenAI chat completions API format
- Returns `choices[0].message.content` with optional citations
- Citations can be in `message.citations`, `message.metadata.citations`, or top-level `citations`

**Frontend (`ChatResponse`)**:
```typescript
interface ChatResponse {
  id?: string;
  model?: string;
  created?: number;
  choices?: ChatChoice[];
  usage?: ChatResponseUsage;
  citations?: ChatCitation[];
}
```

### Improvements Made
1. **Added LocalAI Proxy** - Shell server now proxies `/localai/*` requests
2. **Enhanced Error Handling** - Better error parsing and network error detection
3. **Correct Endpoint** - Changed from `/localai/chat` to `/localai/v1/chat/completions` (OpenAI-compatible)

## AgentFlow/Flows Integration

### Endpoint Configuration
- **UI Endpoint**: `/agentflow/flows` (GET), `/agentflow/flows/{id}/run` (POST)
- **Shell Proxy**: Routes `/agentflow/*` to AgentFlow service (strips `/agentflow` prefix)
- **Backend Service**: AgentFlow FastAPI service on port 9001

### Data Flow
1. UI calls `/agentflow/flows` or `/agentflow/flows/{id}/run`
2. Shell server proxies to `SHELL_AGENTFLOW_ENDPOINT/flows` or `/flows/{id}/run`
3. AgentFlow returns `FlowInfo[]` or `FlowRunResponse`
4. UI displays flows and execution results

### Schema Mapping
**Backend (`FlowInfo`)**:
```python
class FlowInfo:
    local_id: str
    remote_id: Optional[str]
    name: Optional[str]
    description: Optional[str]
    project_id: Optional[str]
    folder_path: Optional[str]
    updated_at: Optional[str]  # ISO datetime string
    synced_at: Optional[str]    # ISO datetime string
```

**Frontend (`FlowInfo`)**:
```typescript
interface FlowInfo {
  local_id: string;
  remote_id?: string | null;
  name?: string | null;
  description?: string | null;
  project_id?: string | null;
  folder_path?: string | null;
  updated_at?: string | null;
  synced_at?: string | null;
}
```

### Improvements Made
1. **Enhanced Error Handling** - Better error parsing for FastAPI responses
2. **Network Error Detection** - Helpful messages for connectivity issues
3. **Empty Response Handling** - Proper handling of empty JSON responses

## Telemetry Integration

### Status
- **Client-side only** - Telemetry is collected in the browser and stored in Zustand state
- **No backend integration** - Currently displays metrics from LocalAI chat interactions
- **Future**: Could be sent to a telemetry service for persistence

### Data Flow
1. LocalAI chat interactions trigger telemetry recording
2. Metrics stored in browser state (Zustand)
3. UI displays real-time metrics from state
4. Metrics reset when user clears history

### Next Steps

1. ✅ ~~Review LocalAI integration~~ - **COMPLETED**
2. ✅ ~~Review AgentFlow/Flows integration~~ - **COMPLETED**
3. ✅ ~~Review Telemetry integration~~ - **COMPLETED** (client-side only)
4. Add integration tests
5. Document all API endpoints and their usage

