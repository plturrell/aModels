<!-- abad5609-dfba-48fe-98d0-9d9b723ee1f2 b1ee9363-4d22-434a-83f2-a38707aaf48b -->
# Build LocalAI Binary and Start All Services

## Overview

Remove the broken `Dockerfile.local`, fix docker-compose.yml to use the correct `Dockerfile` for vaultgemma-server, and create an automated startup script that builds all LocalAI-related services and starts everything without errors or warnings.

## Dockerfile Inventory

The localai directory has 6 Dockerfiles (4 at root, 2 in subdirs):

- `Dockerfile` - vaultgemma-server (main LocalAI) - PRODUCTION ✅
- `Dockerfile.local` - broken, references non-existent LocalAI/ - REMOVE ❌
- `Dockerfile.model-server` - model-server Python service - USED ✅
- `Dockerfile.transformers` - transformers CPU service - USED ✅
- `cmd/config-sync/Dockerfile` - config-sync Go service - USED ✅
- `shim/Dockerfile` - localai-compat shim - USED ✅

## Implementation Steps

### 1. Remove Broken Dockerfile.local

**File:** `services/localai/Dockerfile.local`

- Delete this file completely
- It references a non-existent `LocalAI/` directory that was removed
- It's not used anywhere and causes confusion

### 2. Fix docker-compose.yml to Use Correct Dockerfile

**File:** `infrastructure/docker/brev/docker-compose.yml`

- Change `localai` service dockerfile from `Dockerfile.local` to `Dockerfile`
- Line 117: Change `dockerfile: services/localai/Dockerfile.local` to `dockerfile: services/localai/Dockerfile`
- This will build vaultgemma-server correctly

### 3. Create Build Script for LocalAI

**File:** `scripts/build-localai.sh`

- Build the LocalAI Docker image using `Dockerfile` (vaultgemma-server)
- Check for build errors and warnings
- Verify the binary exists in the built image
- Set proper build context and arguments
- Exit with error code if build fails

### 4. Create Comprehensive Startup Script

**File:** `scripts/start-all-services.sh`

- Stop any running services first (clean slate)
- Build LocalAI image using the build script
- Verify build succeeded with no critical errors
- Start infrastructure services first (redis, postgres, neo4j, elasticsearch, **gitea**)
- Wait for infrastructure to be healthy
- Start core services (model-server, models-storage, transformers, localai, config-sync, localai-compat)
- Wait for core services to be healthy
- Start application services (**catalog** (includes glean), extract, graph, search-inference, etc.)
- Wait for application services to be healthy (especially **catalog**)
- Perform health checks on all services
- Report any errors or warnings
- Exit with appropriate status code

**Service Groups:**
- **Infrastructure:** redis, postgres, neo4j, elasticsearch, **gitea**
- **Core:** model-server, models-storage, transformers, localai, config-sync, localai-compat
- **Application:** **catalog** (includes glean component), extract, graph, search-inference, deepagents

**Gitea Integration:**
- Gitea depends on postgres, so it should start after postgres is healthy
- Gitea health check endpoint: `http://localhost:3000/api/healthz`
- Gitea port: 3000 (HTTP), 2223 (SSH)
- Gitea should be included in infrastructure services group
- Wait for postgres to be healthy before starting gitea
- Health check timeout: 60s (gitea takes longer to initialize)

**Catalog and Glean Integration:**
- Catalog service includes the glean component (glean is a subdirectory within catalog)
- Catalog depends on neo4j, redis, and postgres
- Catalog health check endpoint: `http://localhost:8084/health`
- Catalog port: 8084
- Catalog should be included in application services group
- Wait for neo4j, redis, and postgres to be healthy before starting catalog
- Health check timeout: 60s (catalog needs time to initialize and connect to dependencies)

### 4. Add Error Handling and Validation

- Check Docker is running
- Check required directories exist
- Validate docker-compose.yml syntax
- Capture and report build warnings/errors
- Verify all required volumes are available
- Check port conflicts before starting (including port 3000 for gitea, port 8084 for catalog)

### 5. Integrate with Existing Scripts

- Update `Makefile.services` to use new build/start script
- Ensure compatibility with `scripts/system/start-system.sh`
- Add health check integration

## Files to Modify/Create

1. **Create:** `scripts/build-localai.sh` - LocalAI build script
2. **Create:** `scripts/start-all-services.sh` - Comprehensive startup script  
3. **Modify:** `services/localai/Dockerfile.local` - Fix if LocalAI dir missing (if needed)
4. **Modify:** `Makefile.services` - Add targets for build-localai and start-all

## Success Criteria

- LocalAI binary builds successfully with no errors
- All services start in correct dependency order
- **Gitea starts after postgres and becomes healthy**
- Health checks pass for all services (including gitea)
- No critical warnings in build or startup logs
- Script exits with code 0 on success, non-zero on failure

