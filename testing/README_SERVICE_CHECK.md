# Step 0: Service Health Check

## Overview

**Step 0 is mandatory** - it ensures all required services are running and accessible before running any tests.

## Usage

### Run Service Check Only
```bash
cd /home/aModels
./testing/00_check_services.sh
```

### Run Tests with Service Check
```bash
cd /home/aModels
./testing/run_all_tests_with_check.sh
```

## What It Checks

### 1. Docker Container Status
- ✅ LocalAI container running
- ✅ PostgreSQL container running
- ✅ Redis container running
- ⚠️ Neo4j container (optional)
- ⚠️ Training shell (optional)
- ⚠️ Extract service (optional)
- ⚠️ Elasticsearch (optional)

### 2. Service Accessibility
- ✅ LocalAI health endpoint
- ✅ LocalAI domains endpoint
- ⚠️ Extract service health
- ⚠️ Training service health
- ✅ PostgreSQL connectivity
- ✅ Redis connectivity

## Exit Codes

- **0**: All required services are ready
- **1**: One or more required services are not ready

## Environment Variables Set

After successful check, these are set:
- `LOCALAI_URL` - LocalAI service URL
- `EXTRACT_SERVICE_URL` - Extract service URL
- `TRAINING_SERVICE_URL` - Training service URL

## Required Services

### Critical (Must be running)
- **LocalAI**: For domain detection, inference
- **PostgreSQL**: For metrics, A/B tests
- **Redis**: For caching, traffic splitting

### Optional (Tests will skip if not available)
- Extract Service
- Training Service
- Neo4j
- Elasticsearch

## Troubleshooting

### Services Not Running
```bash
# Start all services
docker compose -f infrastructure/docker/brev/docker-compose.yml up -d

# Wait for services to be ready
sleep 30

# Run service check again
./testing/00_check_services.sh
```

### Services Not Accessible
- Check port mappings: `docker port localai 8080`
- Check firewall settings
- Verify service logs: `docker logs localai`
- Check network connectivity

### LocalAI Not Accessible
```bash
# Check LocalAI logs
docker logs localai

# Restart LocalAI
docker compose -f infrastructure/docker/brev/docker-compose.yml restart localai

# Wait and check again
sleep 20
./testing/00_check_services.sh
```

## Integration with Tests

All test scripts should:
1. Run `00_check_services.sh` first, OR
2. Use `run_all_tests_with_check.sh` which includes the check

This ensures tests only run when services are ready.

## Example Output

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

## Next Steps

After Step 0 passes:
1. Run tests: `./testing/run_all_tests_working.sh`
2. Or run individual test suites
3. All tests will use the environment variables set by Step 0

