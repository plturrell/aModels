# Testing Guide: Error Messages, Health Checks, and Retry Logic

## Overview

This guide covers testing the enhanced error messages, service health checks, and retry logic implemented in the unified search system.

## 1. Testing Enhanced Error Messages

### Prerequisites
- Gateway service running on `http://localhost:8000`
- Search services (inference, extract, catalog) should be **stopped** to test error handling

### Automated Test Script

Run the test script:

```bash
cd services/gateway
python test_search_errors.py
```

### Manual Testing

#### Test 1: All Services Unavailable

```bash
curl -X POST http://localhost:8000/search/unified \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "top_k": 10,
    "sources": ["inference", "knowledge_graph", "catalog"]
  }'
```

**Expected Response:**
- Status: 200 (request succeeds, but sources fail)
- Each source should have an error object with:
  - `error`: Clear error message (e.g., "Connection refused: http://localhost:8090 - Service may not be running (after retries)")
  - `url`: Service URL
  - `type`: Error type ("connection_error", "timeout", or "unknown_error")
- `metadata.sources_failed` should equal the number of unavailable services

#### Test 2: Partial Service Availability

Start one service (e.g., search-inference) and test:

```bash
# Start search-inference service first
# Then run:
curl -X POST http://localhost:8000/search/unified \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "top_k": 10,
    "sources": ["inference", "knowledge_graph", "catalog"]
  }'
```

**Expected Response:**
- `metadata.sources_successful` should be 1
- `metadata.sources_failed` should be 2
- `sources.inference` should contain results
- `sources.knowledge_graph` and `sources.catalog` should contain error objects

### Verification Checklist

- [ ] Error messages are clear and actionable
- [ ] Error messages include service URLs
- [ ] Error types are properly classified
- [ ] Error messages indicate retries were attempted ("after retries")
- [ ] Partial results are returned when some services are available

## 2. Testing Retry Logic

### Prerequisites
- Gateway service running
- Ability to temporarily block/unblock service ports

### Test Scenario: Transient Connection Failure

1. **Start all services**
2. **Temporarily block a service port** (e.g., using `iptables` or firewall)
3. **Make a search request**
4. **Unblock the port during retry attempts**
5. **Verify the request succeeds after retry**

### Manual Test

```bash
# In terminal 1: Block port 8090
sudo iptables -A INPUT -p tcp --dport 8090 -j DROP

# In terminal 2: Make request (will fail initially)
curl -X POST http://localhost:8000/search/unified \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 10, "sources": ["inference"]}'

# In terminal 1: Unblock port 8090 (within retry window)
sudo iptables -D INPUT -p tcp --dport 8090 -j DROP

# Request should succeed after retry
```

### Verification Checklist

- [ ] Retry attempts are logged (check gateway logs)
- [ ] Exponential backoff is used (delays increase: 0.5s, 1s, 2s)
- [ ] Maximum 3 attempts (initial + 2 retries)
- [ ] Only transient errors trigger retries (ConnectError, TimeoutException)
- [ ] HTTP errors (4xx, 5xx) do not trigger retries

## 3. Testing Service Health Checks

### Prerequisites
- Gateway service running
- UI application running

### UI Testing

1. **Navigate to Search Module**
2. **Click on "Health" tab**
3. **Verify health panel displays:**
   - Service status for all services
   - Health count (e.g., "3/10 healthy")
   - Refresh button
   - Last updated timestamp
   - Expandable service details

### API Testing

```bash
curl http://localhost:8000/healthz
```

**Expected Response:**
```json
{
  "gateway": "ok",
  "search_inference": "error:Connection refused",
  "extract": "error:Connection refused",
  "catalog": "error:Connection refused",
  ...
}
```

### Verification Checklist

- [ ] Health endpoint returns status for all services
- [ ] UI displays service health correctly
- [ ] Auto-refresh works (updates every 30 seconds)
- [ ] Manual refresh works
- [ ] Service details are expandable
- [ ] Error messages are shown for failed services
- [ ] Health status is color-coded (green=ok, red=error, yellow=warning)

## 4. Testing Workbench UI Improvements

### Canvas Component

1. **Start workbench application**
2. **Create a new session** (Cmd+K or Ctrl+K)
3. **Verify Canvas displays:**
   - Rich data rendering for objects/arrays
   - Color-coded data types
   - Chip-based key display
   - Session header with timestamp
   - Raw JSON view
   - Empty state with helpful instructions

### Agent Log Panel

1. **Select an active session**
2. **Verify Agent Log Panel displays:**
   - Multiple log entries parsed from session data
   - Log levels (success, error, warning, info, pending)
   - Log types (api, agent, system)
   - Expandable accordion for details
   - Log entry count badge
   - Color-coded indicators

### Verification Checklist

- [ ] Canvas renders nested data structures correctly
- [ ] Canvas shows empty state when no session selected
- [ ] Agent Log Panel parses session data correctly
- [ ] Log entries are expandable
- [ ] Log levels are color-coded
- [ ] Empty states are helpful and informative

## 5. Integration Testing

### Full Workflow Test

1. **Start all services**
2. **Open workbench UI**
3. **Create a search session**
4. **Verify:**
   - Search executes successfully
   - Results appear in Canvas
   - Log entries appear in Agent Log Panel
   - Health status shows all services as healthy

### Error Recovery Test

1. **Start all services**
2. **Make a successful search**
3. **Stop one service** (e.g., search-inference)
4. **Make another search**
5. **Verify:**
   - Search completes with partial results
   - Error messages are clear
   - Health status updates to show service as unavailable
   - UI handles errors gracefully

## 6. Performance Testing

### Retry Performance

- **Test**: Measure time for retry attempts
- **Expected**: Total retry time should be ~1.5s (0.5s + 1s delays)
- **Verify**: Retries don't significantly impact response time

### Health Check Performance

- **Test**: Measure time for health check endpoint
- **Expected**: < 5 seconds for all services
- **Verify**: Health checks don't block main requests

## Troubleshooting

### Common Issues

1. **Services not showing in health check**
   - Verify services are configured in gateway
   - Check service URLs in environment variables
   - Verify services expose `/healthz` endpoint

2. **Retry logic not working**
   - Check gateway logs for retry attempts
   - Verify `retry_utils.py` is imported correctly
   - Check that exceptions are retryable types

3. **UI not updating**
   - Check browser console for errors
   - Verify API endpoints are accessible
   - Check CORS configuration

## Next Steps

- [ ] Add automated integration tests
- [ ] Add performance benchmarks
- [ ] Add monitoring/alerting for service health
- [ ] Add retry metrics to telemetry

