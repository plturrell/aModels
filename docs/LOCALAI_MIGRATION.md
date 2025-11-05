# LocalAI Migration to Official Image

## Overview

Migrated from custom LocalAI server to official LocalAI from [GitHub](https://github.com/mudler/LocalAI).

## Changes Made

### Docker Compose Configuration
- **Before**: Custom build using `services/localai/Dockerfile`
- **After**: Official image `quay.io/go-skynet/local-ai:latest-amd64-cuda`

### Benefits
1. ✅ **Production-ready**: Well-tested, maintained by community
2. ✅ **Better reliability**: Official image is stable and widely used
3. ✅ **Easier maintenance**: No custom build process needed
4. ✅ **Better compatibility**: Works with standard LocalAI configurations
5. ✅ **Active development**: Regular updates and bug fixes

### Configuration Changes

#### Environment Variables
- `MODELS_DIR` → `MODELS_PATH`
- Added `EMBEDDINGS_MODEL=all-MiniLM-L6-v2`
- Added `REDIS_PREFIX=localai`
- Added `PORT=8080`
- Added `THREADS=4` and `CONTEXT_SIZE=512`

#### Volumes
- Models: `/models` (unchanged)
- Config: `/config` (mapped to `services/localai/config`)
- Added `localai-config` volume for LocalAI-specific config

#### Health Check
- Added healthcheck using `/healthz` endpoint
- Standard LocalAI health endpoint

### Image Tags

Available tags based on architecture:
- **x86_64 with CUDA**: `quay.io/go-skynet/local-ai:latest-amd64-cuda`
- **ARM64 with CUDA**: `quay.io/go-skynet/local-ai:latest-aarch64-cuda`
- **CPU only**: `quay.io/go-skynet/local-ai:latest`

### Migration Steps

1. **Stop current LocalAI**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml stop localai
   ```

2. **Pull official image**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml pull localai
   ```

3. **Start with new image**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml up -d localai
   ```

4. **Verify**
   ```bash
   ./testing/00_check_services.sh
   ```

### Configuration Compatibility

The official LocalAI uses YAML configuration files instead of JSON. If you need to migrate:
- Convert `domains.json` to LocalAI YAML format
- Or use LocalAI's built-in domain management
- Or use environment variables for configuration

### References
- Official LocalAI: https://github.com/mudler/LocalAI
- LocalAI Documentation: https://localai.io
- Docker Images: https://quay.io/repository/go-skynet/local-ai

## Next Steps

1. Test health check endpoint
2. Verify models are accessible
3. Test domain configurations
4. Update any custom integrations if needed

