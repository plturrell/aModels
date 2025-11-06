# Missing Services Investigation

## Summary

This document investigates the missing services identified in Week 3 and Week 4 test failures.

## Test Failures Summary

### Week 3 Tests
- **Pattern Learning (0/8 passed)**: All tests fail - modules not found
- **Extraction Intelligence (3/8 passed)**: Pattern transfer modules not found
- **Automation (1/8 passed)**: Orchestration and Analytics services not available

### Week 4 Tests
- **Load Tests**: Training service not available
- **A/B Testing**: A/B test manager module not found

---

## Investigation Results

### 1. Pattern Learning Modules

**Status:** ✅ Code exists, ❌ Not accessible to tests

**Location:** `/home/aModels/services/training/`

**Available Modules:**
- `pattern_learning.py` - Main pattern learning engine
- `pattern_learning_gnn.py` - GNN pattern learner (Phase 7.1)
- `meta_pattern_learner.py` - Meta-pattern learner (Phase 7.2)
- `sequence_pattern_transformer.py` - Sequence pattern transformer
- `active_pattern_learner.py` - Active pattern learner (Phase 7.4)
- `pattern_transfer.py` - Pattern transfer learner

**Issue:** 
- Tests try to import these modules but can't find them
- Python path (`PYTHONPATH`) doesn't include `/workspace/services/training`
- Tests are run from `/workspace/testing` but modules are in `/workspace/services/training`

**Solution:**
1. Update `bootstrap_training_shell.sh` to add `/workspace/services/training` to `PYTHONPATH`
2. Or update tests to add the path dynamically
3. Or create a standalone training service that exposes these as HTTP endpoints

---

### 2. Orchestration Service

**Status:** ✅ Code exists, ❌ No standalone service

**Location:** `/home/aModels/services/orchestration/`

**Available Code:**
- `auto_pipeline.go` - Auto-pipeline orchestrator (Go package)
- `agent_coordinator.go` - Agent coordination (Go package)

**Issue:**
- Tests expect `http://orchestration-service:8080` but no such service exists
- Code exists as Go packages but no `main.go` or Dockerfile
- Orchestration functionality **does exist** in `graph-service` at `/orchestration/process`

**Existing Orchestration:**
- `graph-service` (port 8080) has `/orchestration/process` endpoint
- `gateway` service proxies to `graph-service` for orchestration
- `services/graph/pkg/workflows/orchestration_processor.go` handles orchestration

**Solution Options:**
1. **Option A:** Update tests to use `graph-service:8080/orchestration/process` instead of `orchestration-service:8080`
2. **Option B:** Create a standalone orchestration service by:
   - Adding `main.go` to `services/orchestration/`
   - Creating `Dockerfile` for orchestration service
   - Adding service to `docker-compose.yml`
3. **Option C:** Proxy orchestration through `gateway` service

**Recommendation:** Option A (update tests) - orchestration already exists in graph-service

---

### 3. Analytics Service

**Status:** ✅ Code exists, ❌ No standalone service

**Location:** `/home/aModels/services/analytics/`

**Available Code:**
- `predictive_analytics.go` - Predictive analytics (Go package)
- `recommendation_engine.go` - Recommendation engine (Go package)

**Issue:**
- Tests expect `http://analytics-service:8080` but no such service exists
- Code exists as Go packages but no `main.go` or Dockerfile
- Analytics functionality **does exist** in:
  - `catalog` service (has analytics dashboard)
  - `postgres` service (has analytics endpoints via gRPC)

**Existing Analytics:**
- `catalog` service has analytics dashboard at `/analytics`
- `postgres` service has analytics via gRPC (`GetAnalytics` method)
- `services/postgres/gateway/app.py` exposes analytics via HTTP

**Solution Options:**
1. **Option A:** Update tests to use `catalog:8084/analytics` or `postgres` service analytics
2. **Option B:** Create a standalone analytics service by:
   - Adding `main.go` to `services/analytics/`
   - Creating `Dockerfile` for analytics service
   - Adding service to `docker-compose.yml`
3. **Option C:** Use existing analytics in catalog/postgres services

**Recommendation:** Option A (update tests) - analytics already exists in catalog/postgres

---

### 4. Training Service

**Status:** ✅ Code exists, ❌ No standalone service

**Location:** `/home/aModels/services/training/`

**Available Code:**
- Python modules for training pipeline, pattern learning, domain training, etc.
- No `main.py` or `Dockerfile` for a standalone service
- Training functionality is library code, not a service

**Issue:**
- Tests expect `http://training-service:8080` but no such service exists
- Training modules are Python libraries, not HTTP services
- `training-shell` container exists but it's just a shell, not a service

**Solution Options:**
1. **Option A:** Create a training service by:
   - Adding `main.py` with FastAPI/Flask server
   - Creating `Dockerfile` for training service
   - Adding service to `docker-compose.yml`
   - Exposing training endpoints as HTTP API
2. **Option B:** Update tests to import training modules directly (not via HTTP)
3. **Option C:** Use `training-shell` container and run training via Python scripts

**Recommendation:** Option A (create training service) - tests expect HTTP endpoints

---

### 5. A/B Test Manager

**Status:** ✅ Code exists, ❌ Not a service

**Location:** `/home/aModels/services/training/ab_testing.py`

**Issue:**
- Tests try to import `ABTestManager` but can't find it
- Module exists but Python path doesn't include it
- It's a Python class, not a service

**Solution:**
- Update `PYTHONPATH` to include `/workspace/services/training`
- Or create a training service that exposes A/B testing endpoints

---

## Current Docker Compose Services

**Existing Services in `docker-compose.yml`:**
- ✅ `localai` - LocalAI inference engine
- ✅ `localai-compat` - LocalAI compatibility shim
- ✅ `extract-service` - Extract service (Go)
- ✅ `graph-server` - Graph service with orchestration endpoints
- ✅ `deepagents` - DeepAgents service
- ✅ `catalog` - Catalog service (has analytics)
- ✅ `postgres` - PostgreSQL with analytics gRPC
- ✅ `redis` - Redis cache
- ✅ `neo4j` - Neo4j graph database
- ✅ `elasticsearch` - Elasticsearch
- ✅ `training-shell` - Training shell (not a service, just a container)
- ❌ `orchestration-service` - **NOT DEFINED**
- ❌ `analytics-service` - **NOT DEFINED**
- ❌ `training-service` - **NOT DEFINED**

---

## Recommended Solutions

### Immediate Fixes (Update Tests)

1. **Update Pattern Learning Tests:**
   - Add `/workspace/services/training` to `PYTHONPATH` in `bootstrap_training_shell.sh`
   - Update tests to import from `services.training` modules

2. **Update Orchestration Tests:**
   - Change `ORCHESTRATION_SERVICE_URL` from `http://orchestration-service:8080` to `http://graph-server:8080/orchestration/process`
   - Or use `http://gateway:8000/orchestration/process` if gateway is running

3. **Update Analytics Tests:**
   - Change `ANALYTICS_SERVICE_URL` from `http://analytics-service:8080` to:
     - `http://catalog:8084/analytics` (if catalog is running)
     - Or use postgres analytics gRPC endpoint

4. **Update Training Service Tests:**
   - Create a training service (see below)
   - Or update tests to import training modules directly (not via HTTP)

### Long-term Solutions (Create Services)

1. **Create Training Service:**
   - Add `main.py` with FastAPI server
   - Expose endpoints: `/health`, `/train`, `/patterns/learn`, etc.
   - Create `Dockerfile`
   - Add to `docker-compose.yml`

2. **Optional: Create Standalone Orchestration Service:**
   - Only if we want to separate orchestration from graph-service
   - Add `main.go` with HTTP server
   - Create `Dockerfile`
   - Add to `docker-compose.yml`

3. **Optional: Create Standalone Analytics Service:**
   - Only if we want to separate analytics from catalog/postgres
   - Add `main.go` with HTTP server
   - Create `Dockerfile`
   - Add to `docker-compose.yml`

---

## Implementation Plan

### Phase 1: Quick Fixes (Update Tests)
1. ✅ Update `bootstrap_training_shell.sh` to add training service to `PYTHONPATH`
2. ✅ Update orchestration tests to use `graph-server:8080/orchestration/process`
3. ✅ Update analytics tests to use `catalog:8084/analytics`
4. ✅ Update pattern learning tests to import from correct path

### Phase 2: Create Training Service (Recommended)
1. Create `services/training/main.py` with FastAPI server
2. Create `services/training/Dockerfile`
3. Add `training-service` to `docker-compose.yml`
4. Expose endpoints:
   - `/health`
   - `/train`
   - `/patterns/learn`
   - `/patterns/gnn`
   - `/patterns/meta`
   - `/ab-test/create`
   - `/ab-test/route`

### Phase 3: Optional Services
- Create standalone orchestration service (if needed)
- Create standalone analytics service (if needed)

---

## Files to Update

### Test Configuration
- `testing/bootstrap_training_shell.sh` - Add `PYTHONPATH` for training modules
- `testing/test_pattern_learning.py` - Update imports
- `testing/test_extraction_intelligence.py` - Update pattern transfer imports
- `testing/test_automation.py` - Update orchestration/analytics URLs
- `testing/test_load.py` - Update training service URL

### Docker Compose
- `infrastructure/docker/brev/docker-compose.yml` - Add training-service (and optional orchestration/analytics)

### New Service Files
- `services/training/main.py` - FastAPI server for training service
- `services/training/Dockerfile` - Dockerfile for training service
- `services/training/requirements-service.txt` - Service dependencies (if different from training modules)

---

## Status

**Investigation Complete:** ✅  
**Quick Fixes Identified:** ✅  
**Long-term Solutions Identified:** ✅  
**Ready for Implementation:** ✅

---

**Next Steps:**
1. Implement quick fixes (update tests and PYTHONPATH)
2. Create training service
3. Re-run Week 3 and Week 4 tests
4. Document any remaining issues

