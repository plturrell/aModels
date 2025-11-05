# Step 0: Service Health Check - Complete ✅

## Overview

**Step 0 is now the mandatory starting point** for all test execution in the aModels platform.

## What Was Created

### 1. Main Service Check Script
- **File**: `testing/00_check_services.sh`
- **Purpose**: Comprehensive service health check
- **Features**:
  - Checks Docker container status
  - Verifies service accessibility
  - Sets environment variables
  - Provides clear pass/fail status

### 2. Combined Test Runner
- **File**: `testing/run_all_tests_with_check.sh`
- **Purpose**: Runs Step 0 first, then all tests
- **Benefit**: Single command to check services and run tests

### 3. Documentation
- `testing/README_SERVICE_CHECK.md` - Detailed service check guide
- `docs/STEP_0_SERVICE_CHECK.md` - Complete Step 0 documentation
- `testing/STEP_0_README.txt` - Quick reference

### 4. Updated Test Runner
- **File**: `testing/run_all_tests_working.sh`
- **Update**: Added warning to run Step 0 first

## How Step 0 Works

### Step 1: Docker Container Status
Checks if required containers are running:
- ✅ LocalAI (required)
- ✅ PostgreSQL (required)
- ✅ Redis (required)
- ⚠️ Optional services

### Step 2: Service Accessibility
Verifies services are actually accessible:
- ✅ LocalAI health endpoint
- ✅ LocalAI domains endpoint
- ✅ PostgreSQL connectivity
- ✅ Redis connectivity
- ⚠️ Optional services

### Step 3: Environment Setup
Sets environment variables based on actual service locations:
- `LOCALAI_URL` - Detected from port mapping
- `EXTRACT_SERVICE_URL` - Detected from port mapping
- `TRAINING_SERVICE_URL` - Detected from port mapping

### Step 4: Exit Status
- **Exit 0**: All required services ready → Tests can proceed
- **Exit 1**: Services not ready → Fix services first

## Usage

### Method 1: Manual Step 0
```bash
# Step 0: Check services
cd /home/aModels
./testing/00_check_services.sh

# If Step 0 passes, run tests
./testing/run_all_tests_working.sh
```

### Method 2: Automatic Step 0
```bash
# Combined: Step 0 + All Tests
cd /home/aModels
./testing/run_all_tests_with_check.sh
```

## Step 0 Output

### Success Example
```
============================================================
Step 0: Service Health Check
============================================================
Verifying all required services are running and accessible...

============================================================
Step 1: Docker Container Status
============================================================
Checking localai... ✅ Running
Checking postgres... ✅ Running
Checking redis... ✅ Running

============================================================
Step 2: Service Accessibility
============================================================
Checking LocalAI Health... ✅ Ready
Checking LocalAI Domains endpoint... ✅ Ready (5 domains found)
Checking PostgreSQL... ✅ Ready
Checking Redis... ✅ Ready

============================================================
Service Health Check Summary
============================================================
Total Services Checked: 6
✅ Ready: 6
❌ Failed: 0

============================================================
✅ ALL REQUIRED SERVICES ARE READY
============================================================
```

### Failure Example
```
============================================================
❌ SOME REQUIRED SERVICES ARE NOT READY
============================================================

Please ensure all required services are running:
  docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
```

## Required Services

### Critical (Must Pass)
- **LocalAI**: Domain detection, inference
- **PostgreSQL**: Metrics, A/B tests
- **Redis**: Caching, traffic splitting

### Optional (Tests Skip If Not Available)
- Extract Service
- Training Service
- Neo4j
- Elasticsearch

## Benefits

1. **Prevents Test Failures**: Ensures services are ready before tests
2. **Clear Feedback**: Shows exactly which services are ready/not ready
3. **Automatic Configuration**: Sets correct environment variables
4. **Consistent Environment**: All tests use same service URLs
5. **Early Detection**: Catches service issues before running tests

## Integration

Step 0 is now integrated into the test workflow:
- All test documentation references Step 0
- Test runners check for Step 0
- Combined script includes Step 0 automatically

## Next Steps

1. **Always run Step 0 first** before any test execution
2. **Fix service issues** if Step 0 fails
3. **Use environment variables** set by Step 0
4. **Re-run Step 0** if services are restarted

## Files Created

- ✅ `testing/00_check_services.sh` - Main service check script
- ✅ `testing/run_all_tests_with_check.sh` - Combined runner
- ✅ `testing/README_SERVICE_CHECK.md` - Service check guide
- ✅ `docs/STEP_0_SERVICE_CHECK.md` - Complete documentation
- ✅ `testing/STEP_0_README.txt` - Quick reference

## Status

✅ **Step 0 is complete and ready to use**

This is now the mandatory starting point for all test execution in the aModels platform.

---

**Created**: 2025-01-XX  
**Status**: Complete ✅  
**Next**: Run Step 0 before executing any tests

