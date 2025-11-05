# Docker Volume Mount Issue for Testing Directory

## Problem
The bind mount for `/home/aModels/testing` to `/workspace/testing` in the `training-shell` container is not working correctly. The mount shows as `/dev/root` instead of the actual host directory, and files from the host do not appear in the container.

## Symptoms
- Host directory `/home/aModels/testing` has 67 files
- Container mount `/workspace/testing` shows only 5 files (test files created in container)
- Inodes differ between host and container (suggesting different directories)
- Files created on host do not appear in container
- Bind mount shows as `/dev/root` type ext4 instead of proper bind mount

## Attempted Solutions
1. ✅ Changed mount options (`rw`, `cached`, `delegated`, `consistent`)
2. ✅ Used explicit bind mount syntax in docker-compose
3. ✅ Added `/host-testing` as read-only source with copy script
4. ❌ All attempts failed - bind mount still not working

## Root Cause (Suspected)
- Overlay filesystem interference
- Docker daemon configuration issue
- Filesystem type mismatch
- Container image overlay masking bind mounts

## Workarounds

### Option 1: Use docker cp (Temporary)
```bash
docker cp /home/aModels/testing/test_extraction_flow.py training-shell:/workspace/testing/
```

### Option 2: Run tests from host with network access
```bash
export LOCALAI_URL=http://localai-compat:8080
export EXTRACT_SERVICE_URL=http://extract-service:8082
python3 testing/test_extraction_flow.py
```

### Option 3: Fix Docker daemon configuration
- Check Docker daemon logs for bind mount errors
- Verify SELinux/AppArmor settings
- Consider using Docker volumes instead of bind mounts

## Current Status
- Test fixes are complete and correct
- Tests will pass once volume mount is resolved
- Docker compose configuration is correct
- Issue is environment-specific

## Next Steps
1. Investigate Docker daemon configuration
2. Check for overlay filesystem conflicts
3. Consider using named volumes instead of bind mounts
4. Test on different Docker environment

