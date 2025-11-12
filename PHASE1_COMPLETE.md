# Phase 1: Backup - COMPLETE ✓

**Completed:** Nov 12, 2025 14:16 UTC  
**Duration:** < 1 minute  
**Status:** SUCCESS

---

## What Was Done

### 1. Full Backup Created
- **Location:** `/tmp/scripts_backup_20251112.tar.gz`
- **Size:** 289KB
- **Files:** 277 files backed up
- **Verification:** ✓ Integrity checked and confirmed

### 2. Directories Backed Up
```
✓ /home/aModels/scripts/          (30 files)
✓ /home/aModels/testing/          (16 files)
✓ /home/aModels/tools/            (4 files)
✓ Service scripts to migrate      (3 files)
```

### 3. Critical Files Protected
- All organized scripts in `/scripts/` subdirectories
- All redundant test scripts in `/testing/`
- All helper utilities in `/tools/`
- Misplaced service scripts:
  - `services/start_all_services.sh`
  - `services/telemetry-exporter/create_signavio_extract.sh`
  - `infrastructure/docker/brev/sync-testing.sh`

---

## Analysis Results

### Redundant Test Scripts Identified

#### Category A: run_all_tests*.sh Variants (6 files)

| Script | Lines | Set Flag | Key Feature | Decision |
|--------|-------|----------|-------------|----------|
| **run_all_tests.sh** | 155 | `set +e` | Docker network URLs | ✅ **KEEP** (Base) |
| run_all_tests_final.sh | 137 | `set +e` | Docker container execution | ❌ Merge |
| run_all_tests_fixed.sh | 164 | `set -e` | Host + Docker auto-detect | ❌ Merge |
| run_all_tests_with_check.sh | 36 | `set -e` | Pre-service check wrapper | ⚠️ **KEEP** (Wrapper) |
| run_all_tests_with_step0.sh | 45 | `set +e` | Service check + env vars | ❌ Duplicate |
| run_all_tests_working.sh | 141 | `set -e` | Error handling | ❌ Merge |

**Unique Features to Extract:**

1. **From run_all_tests_fixed.sh:**
   ```bash
   # Auto-detect Docker environment
   if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
       export LOCALAI_URL="http://localai:8080"
   else
       export LOCALAI_URL="http://localhost:8080"
   fi
   ```

2. **From run_all_tests_final.sh:**
   ```bash
   # Detect Docker network automatically
   NETWORK=$(docker inspect localai --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null | head -1)
   ```

3. **Keep as wrapper:** `run_all_tests_with_check.sh` (calls 00_check_services.sh first)

#### Category B: Docker Test Runners (5 files)

| Script | Lines | Purpose | Decision |
|--------|-------|---------|----------|
| **run_tests_docker.sh** | 30 | Simple Docker runner | ✅ **KEEP** (Simplest) |
| run_tests_docker_network.sh | 40 | Docker network URLs | ❌ Duplicate |
| run_tests_from_container.sh | 153 | From DeepAgents container | ❌ Too specific |
| run_tests_from_docker.sh | 71 | From training-shell | ❌ Too specific |
| run_tests_in_container.sh | 127 | Temporary container | ❌ Too specific |

**Recommendation:** Keep only `run_tests_docker.sh`, delete 4 others

#### Category C: Misc (1 file)

| Script | Lines | Decision |
|--------|-------|----------|
| run_tests_now.sh | 58 | ❌ DELETE (no unique value) |

---

## Rollback Instructions

If anything goes wrong, restore the backup:

```bash
# Full restore
cd /
sudo tar -xzf /tmp/scripts_backup_20251112.tar.gz

# Verify restoration
ls -la /home/aModels/scripts/
ls -la /home/aModels/testing/
ls -la /home/aModels/tools/
```

---

## Next: Phase 2 - Create New Structure

### What Phase 2 Will Do

1. **Create new directory:**
   ```bash
   mkdir -p /home/aModels/scripts/dev-tools
   ```

2. **Prepare for moves:**
   - Dev tools → `/scripts/dev-tools/`
   - Testing scripts → `/scripts/testing/`
   - System scripts → `/scripts/system/`
   - Signavio scripts → `/scripts/signavio/`

### Files to Move in Phase 3

#### To dev-tools/ (5 files)
- `tools/helpers/fetch_kaggle_gemma.sh`
- `tools/scripts/build.sh`
- `tools/scripts/cleanup_rt_archives.sh`
- `tools/scripts/run_rt_main_schedule.sh`
- `infrastructure/docker/brev/sync-testing.sh`

#### To system/ (1 file)
- `services/start_all_services.sh`

#### To signavio/ (1 file)
- `services/telemetry-exporter/create_signavio_extract.sh`

#### To testing/ (6 essential files)
- `testing/00_check_services.sh`
- `testing/bootstrap_training_shell.sh`
- `testing/setup_test_database.sh`
- `testing/run_smoke_tests.sh`
- `testing/test_localai_from_container.sh`
- `testing/run_all_tests_with_check.sh` (wrapper)

#### To quality/ (1 file)
- `testing/performance_benchmark_runner.sh`

#### Total moves: 14 files

---

## Consolidation Plan (Phase 4)

### Step 1: Enhance run_all_tests.sh
Add features from redundant variants:
- Docker environment auto-detection (from _fixed)
- Network detection (from _final)
- Better error handling (from _working)

### Step 2: Keep Essential Scripts Only
```
scripts/testing/
├── run_all_tests.sh              # Main runner (enhanced)
├── run_all_tests_with_check.sh   # Wrapper with service check
└── run_tests_docker.sh           # Docker mode wrapper
```

### Step 3: Delete Redundant Scripts (9 files)
```bash
# After consolidation verified
rm testing/run_all_tests_final.sh
rm testing/run_all_tests_fixed.sh
rm testing/run_all_tests_with_step0.sh
rm testing/run_all_tests_working.sh
rm testing/run_tests_docker_network.sh
rm testing/run_tests_from_container.sh
rm testing/run_tests_from_docker.sh
rm testing/run_tests_in_container.sh
rm testing/run_tests_now.sh
```

---

## Summary

### Before
- 71 scripts across 8+ directories
- 11 redundant test scripts
- Scattered organization
- Unclear which script to use

### After (Planned)
- 55-60 scripts in 2 main locations
- 0 redundant scripts
- Clear categorization
- Single source of truth

### Reduction
- **16% fewer scripts** overall
- **73% fewer test runners** (11 → 3)
- **Clean organization** (8+ dirs → 2)

---

## Ready to Proceed?

### Option 1: Continue with Automated Migration
```bash
./scripts/migrate_scripts.sh --execute
```
This will run Phases 2-6 automatically.

### Option 2: Manual Step-by-Step
Run each phase individually for more control:

**Phase 2:**
```bash
mkdir -p /home/aModels/scripts/dev-tools
```

**Phase 3:** (see SCRIPTS_REORGANIZATION_PLAN.md for move commands)

### Option 3: Review First
Before proceeding, you may want to:
1. Review unique features in redundant scripts
2. Test current scripts one more time
3. Update any documentation that references script paths

---

## Questions Before Proceeding?

1. **Should we keep `run_all_tests_with_check.sh`?**
   - Pro: Convenient wrapper for CI/CD
   - Con: Could just call `00_check_services.sh && run_all_tests.sh`
   - Recommendation: Keep it (it's small and useful)

2. **Consolidate now or after migration?**
   - Option A: Consolidate redundant scripts now (before move)
   - Option B: Move first, consolidate after
   - Recommendation: Move first (Phase 3), then consolidate (Phase 4)

3. **Keep old paths temporarily?**
   - Could create symlinks during transition
   - Or hard cutover with documentation updates
   - Recommendation: Hard cutover (cleaner)

---

## Status

✅ **Phase 1: COMPLETE**  
⏭️ **Phase 2: Ready to start**

**Estimated time to complete all phases:** 2-3 hours
