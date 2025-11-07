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

### Next Steps

1. Review LocalAI integration
2. Review AgentFlow/Flows integration
3. Review Telemetry integration
4. Add integration tests
5. Document all API endpoints and their usage

