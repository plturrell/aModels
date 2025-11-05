# Testing Directory Mount Setup

## Current Solution

The `training-shell` container uses a **named Docker volume** (`testing-files`) for `/workspace/testing` instead of a direct bind mount due to overlay filesystem issues.

## How to Populate Testing Files

### Option 1: Use docker cp (Recommended)
```bash
docker cp /home/aModels/testing/. training-shell:/workspace/testing/
```

### Option 2: Automatic Sync Script
Create a helper script to sync files:
```bash
#!/bin/bash
# sync-testing.sh
docker cp /home/aModels/testing/. training-shell:/workspace/testing/
echo "Testing files synced to container"
```

### Option 3: Run tests from host
```bash
export LOCALAI_URL=http://localai-compat:8080
export EXTRACT_SERVICE_URL=http://extract-service:8082
python3 testing/test_extraction_flow.py
```

## Why Named Volume?

The bind mount `/home/aModels/testing:/workspace/testing` was showing as `/dev/root` instead of the actual host directory due to:
- Overlay filesystem interference
- Docker bind mount limitations in this environment
- Filesystem type compatibility issues

## Container Startup

The container will check if `/workspace/testing` is empty and warn if files need to be synced.

## Updating Files

After making changes to test files, run:
```bash
docker cp /home/aModels/testing/. training-shell:/workspace/testing/
```

Or restart the container and it will preserve files in the named volume.

