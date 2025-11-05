# Step 0: Service Health Check - Mandatory First Step

## Overview

**Step 0 is the mandatory starting point** for all test execution. It ensures all required services are running and accessible before any tests are run.

## Why Step 0?

- **Prevents test failures** due to missing services
- **Sets correct environment variables** based on actual service locations
- **Provides clear feedback** on which services are ready
- **Ensures tests run in a consistent environment**

## Quick Start

### Run Step 0 Only
```bash
cd /home/aModels
./testing/00_check_services.sh
```

### Run Step 0 + All Tests
```bash
cd /home/aModels
./testing/run_all_tests_with_check.sh
```

## What Step 0 Checks

### Docker Container Status
- ✅ LocalAI (required)
- ✅ PostgreSQL (required)
- ✅ Redis (required)
- ⚠️ Neo4j (optional)
- ⚠️ Training shell (optional)
- ⚠️ Extract service (optional)
- ⚠️ Elasticsearch (optional)

### Service Accessibility
- ✅ LocalAI health endpoint
- ✅ LocalAI domains endpoint
- ✅ PostgreSQL connectivity
- ✅ Redis connectivity
- ⚠️ Extract service (if available)
- ⚠️ Training service (if available)

## Exit Codes

- **0**: All required services ready → Tests can proceed
- **1**: Services not ready → Fix services before running tests

## Environment Variables Set

After Step 0 passes, these environment variables are automatically set:
- `LOCALAI_URL` - LocalAI service URL (with correct port)
- `EXTRACT_SERVICE_URL` - Extract service URL
- `TRAINING_SERVICE_URL` - Training service URL

## Example Output

### Success (All Services Ready)
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

Environment variables set:
  export LOCALAI_URL="http://localhost:8081"
  export EXTRACT_SERVICE_URL="http://localhost:19080"
  export TRAINING_SERVICE_URL="http://localhost:8080"

You can now run tests:
  ./testing/run_all_tests_working.sh
```

### Failure (Services Not Ready)
```
============================================================
❌ SOME REQUIRED SERVICES ARE NOT READY
============================================================

Please ensure all required services are running:
  docker compose -f infrastructure/docker/brev/docker-compose.yml up -d

Then wait for services to be ready and run this check again:
  ./testing/00_check_services.sh
```

## Troubleshooting

### Services Not Running
```bash
# Start all services
docker compose -f infrastructure/docker/brev/docker-compose.yml up -d

# Wait for services to initialize
sleep 30

# Run Step 0 again
./testing/00_check_services.sh
```

### LocalAI Not Accessible
```bash
# Check LocalAI logs
docker logs localai

# Restart LocalAI
docker compose -f infrastructure/docker/brev/docker-compose.yml restart localai

# Wait and check
sleep 20
./testing/00_check_services.sh
```

### Port Mapping Issues
```bash
# Check port mappings
docker port localai 8080

# Verify port is accessible
curl http://localhost:8081/health
```

## Integration with Test Workflow

### Option 1: Manual Step 0
```bash
# Step 0: Check services
./testing/00_check_services.sh

# If Step 0 passes, run tests
./testing/run_all_tests_working.sh
```

### Option 2: Automatic Step 0
```bash
# Combined script runs Step 0 first, then tests
./testing/run_all_tests_with_check.sh
```

## Best Practices

1. **Always run Step 0 first** before running any tests
2. **Fix service issues** if Step 0 fails
3. **Use environment variables** set by Step 0 in your tests
4. **Re-run Step 0** if services are restarted
5. **Check service logs** if Step 0 fails

## Required vs Optional Services

### Required Services (Step 0 will fail if not ready)
- LocalAI
- PostgreSQL
- Redis

### Optional Services (Tests will skip if not available)
- Extract Service
- Training Service
- Neo4j
- Elasticsearch

## Next Steps

After Step 0 passes:
1. ✅ Environment variables are set
2. ✅ Services are verified accessible
3. ✅ You can proceed with test execution

Run tests:
```bash
./testing/run_all_tests_working.sh
```

---

**Remember**: Step 0 is the mandatory starting point. Always run it first!

