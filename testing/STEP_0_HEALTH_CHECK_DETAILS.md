# Step 0: Health Check Details

## What Step 0 Actually Checks

Step 0 verifies services are **Running, Reachable, and Healthy**:

### 1. Running Check ✅
**What it checks**: Docker container status
**How it works**:
- Uses `docker compose ps` to check if container is "Up"
- Also checks Docker health status (healthy/unhealthy) if available
- Verifies container is actually running, not just created

**Example**:
```bash
docker compose ps localai | grep "Up"
# Result: ✅ Running (if container is up)
```

### 2. Reachable Check ✅
**What it checks**: Network connectivity
**How it works**:
- Makes HTTP request to service endpoint
- Tests connection (not just DNS lookup)
- Verifies port is accessible and service is listening

**Example**:
```python
httpx.get('http://localhost:8081/health', timeout=10)
# Result: ✅ Reachable (if connection succeeds)
#         ❌ Not Reachable (if connection refused/timeout)
```

### 3. Healthy Check ✅
**What it checks**: Service health status
**How it works**:
- Checks HTTP status code is 200
- For `/health` endpoints, also checks response content:
  - Looks for "ok", "healthy", "up" in response
  - Checks JSON health status if applicable
- Verifies service is actually functioning, not just running

**Example**:
```python
response = httpx.get('http://localhost:8081/health')
# Check 1: Status code == 200
# Check 2: Response contains health indicators
# Result: ✅ Healthy (if both pass)
```

## Three-Level Verification

### Level 1: Running (Container Status)
```
✅ Container is "Up" in Docker
✅ Container health status (if available)
```

### Level 2: Reachable (Network Connectivity)
```
✅ Can connect to service port
✅ HTTP request succeeds
✅ No connection refused/timeout errors
```

### Level 3: Healthy (Service Functionality)
```
✅ HTTP status code is 200
✅ Health endpoint returns healthy status
✅ Service is functional (not just running)
```

## What Each Check Catches

### Running Check Catches:
- ❌ Container not started
- ❌ Container crashed
- ❌ Container in unhealthy state (Docker healthcheck)

### Reachable Check Catches:
- ❌ Port not mapped correctly
- ❌ Firewall blocking connection
- ❌ Service not listening on port
- ❌ Network connectivity issues

### Healthy Check Catches:
- ❌ Service crashed after starting
- ❌ Service not initialized properly
- ❌ Service returning error status
- ❌ Service unhealthy (but running)

## Example Output

```
============================================================
Step 1: Docker Container Status
============================================================
Checking localai (Docker)... ✅ Running & Healthy
Checking postgres (Docker)... ✅ Running & Healthy
Checking redis (Docker)... ✅ Running

============================================================
Step 2: Service Accessibility
============================================================
Checking LocalAI service (Running, Reachable & Healthy)...
Checking LocalAI Health... ✅ Running, Reachable & Healthy
Checking LocalAI Domains endpoint... ✅ Ready (5 domains found)
Checking PostgreSQL... ✅ Ready
Checking Redis... ✅ Ready
```

## Failure Scenarios

### Scenario 1: Container Not Running
```
Checking localai (Docker)... ❌ Not Running (REQUIRED)
```
**Fix**: Start container with `docker compose up -d localai`

### Scenario 2: Container Running But Not Reachable
```
Checking localai (Docker)... ✅ Running
Checking LocalAI Health... ❌ Not Ready (Connection refused)
```
**Fix**: Check port mapping, firewall, or service logs

### Scenario 3: Container Running But Unhealthy
```
Checking localai (Docker)... ⚠️  Running but Unhealthy
Checking LocalAI Health... ❌ Not Ready (Status 500)
```
**Fix**: Check service logs, verify configuration, check dependencies

## Benefits of Three-Level Check

1. **Early Detection**: Catches issues at container level
2. **Network Verification**: Ensures services are accessible
3. **Functional Verification**: Confirms services are actually working
4. **Clear Feedback**: Shows exactly which level failed

## Summary

Step 0 performs **comprehensive health checking**:
- ✅ **Running**: Container status check
- ✅ **Reachable**: Network connectivity check  
- ✅ **Healthy**: Service functionality check

All three levels must pass for a service to be considered ready.

