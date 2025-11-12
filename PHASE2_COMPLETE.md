# Phase 2: Create Directory Structure - COMPLETE ✓

**Completed:** Nov 12, 2025 14:22 UTC  
**Duration:** < 1 minute  
**Status:** SUCCESS

---

## What Was Done

### 1. Created New Directory
```bash
✓ /home/aModels/scripts/dev-tools/
```

### 2. Verified Complete Structure

```
/home/aModels/scripts/
├── data/           (11 .sh files) ✓ Existing
├── dev-tools/      (0 .sh files)  ✨ NEW - Ready for files
├── lib/            (1 .sh file)   ✓ Existing
├── quality/        (5 .sh files)  ✓ Existing
├── signavio/       (2 .sh files)  ✓ Existing - Will receive 1 more
├── system/         (7 .sh files)  ✓ Existing - Will receive 1 more
├── templates/                     ✓ Existing
└── testing/        (4 .sh files)  ✓ Existing - Will receive 6 more
```

**Total:** 8 organized categories ready

---

## Files Verified for Phase 3 Migration

### ✅ All Source Files Found (14 files)

#### To dev-tools/ (5 files)
- ✓ `tools/helpers/fetch_kaggle_gemma.sh`
- ✓ `tools/scripts/build.sh`
- ✓ `tools/scripts/cleanup_rt_archives.sh`
- ✓ `tools/scripts/run_rt_main_schedule.sh`
- ✓ `infrastructure/docker/brev/sync-testing.sh`

#### To system/ (1 file)
- ✓ `services/start_all_services.sh`

#### To signavio/ (1 file)
- ✓ `services/telemetry-exporter/create_signavio_extract.sh`

#### To testing/ (6 files)
- ✓ `testing/00_check_services.sh`
- ✓ `testing/bootstrap_training_shell.sh`
- ✓ `testing/setup_test_database.sh`
- ✓ `testing/run_smoke_tests.sh`
- ✓ `testing/test_localai_from_container.sh`
- ✓ `testing/run_all_tests_with_check.sh`

#### To quality/ (1 file)
- ✓ `testing/performance_benchmark_runner.sh`

**Total Ready to Move:** 14 files

---

## Current State

### Scripts Distribution (Before Phase 3)

| Directory | Current Files | Will Receive | Total After |
|-----------|---------------|--------------|-------------|
| scripts/data/ | 11 | 0 | 11 |
| scripts/dev-tools/ | 0 | +5 | 5 |
| scripts/lib/ | 1 | 0 | 1 |
| scripts/quality/ | 5 | +1 | 6 |
| scripts/signavio/ | 2 | +1 | 3 |
| scripts/system/ | 7 | +1 | 8 |
| scripts/testing/ | 4 | +6 | 10 |
| **Total** | **30** | **+14** | **44** |

### Source Directories (To Be Cleaned)

| Directory | Files | Status |
|-----------|-------|--------|
| /testing/ | 18 | Will move 7, analyze 11 redundant |
| /tools/ | 4 | Will move all 4 |
| /services/ (root) | 2 | Will move both |
| /infrastructure/ | 1 | Will move 1 |

---

## Next: Phase 3 - Move Files

### What Phase 3 Will Do

**Safe operations only:**
1. Move 14 files to their new locations
2. Preserve file permissions and timestamps
3. No deletions yet (redundant scripts stay for now)

### Move Commands Preview

```bash
# Dev Tools (5 moves)
mv /home/aModels/tools/helpers/fetch_kaggle_gemma.sh \
   /home/aModels/scripts/dev-tools/

mv /home/aModels/tools/scripts/build.sh \
   /home/aModels/scripts/dev-tools/

mv /home/aModels/tools/scripts/cleanup_rt_archives.sh \
   /home/aModels/scripts/dev-tools/

mv /home/aModels/tools/scripts/run_rt_main_schedule.sh \
   /home/aModels/scripts/dev-tools/

mv /home/aModels/infrastructure/docker/brev/sync-testing.sh \
   /home/aModels/scripts/dev-tools/

# System (1 move)
mv /home/aModels/services/start_all_services.sh \
   /home/aModels/scripts/system/

# Signavio (1 move)
mv /home/aModels/services/telemetry-exporter/create_signavio_extract.sh \
   /home/aModels/scripts/signavio/

# Testing (6 moves)
mv /home/aModels/testing/00_check_services.sh \
   /home/aModels/scripts/testing/

mv /home/aModels/testing/bootstrap_training_shell.sh \
   /home/aModels/scripts/testing/

mv /home/aModels/testing/setup_test_database.sh \
   /home/aModels/scripts/testing/

mv /home/aModels/testing/run_smoke_tests.sh \
   /home/aModels/scripts/testing/

mv /home/aModels/testing/test_localai_from_container.sh \
   /home/aModels/scripts/testing/

mv /home/aModels/testing/run_all_tests_with_check.sh \
   /home/aModels/scripts/testing/

# Quality (1 move)
mv /home/aModels/testing/performance_benchmark_runner.sh \
   /home/aModels/scripts/quality/
```

**Total:** 14 mv operations

---

## Redundant Scripts Remaining in /testing/

These will be analyzed in Phase 4 (not moved yet):

### Keep for Analysis (3 files)
- `testing/run_all_tests.sh` - Main runner (to enhance)
- `testing/run_tests_docker.sh` - Docker wrapper (to keep)
- `testing/run_tests_now.sh` - May have unique features?

### Likely Redundant (8 files)
- `testing/run_all_tests_final.sh`
- `testing/run_all_tests_fixed.sh`
- `testing/run_all_tests_with_step0.sh`
- `testing/run_all_tests_working.sh`
- `testing/run_tests_docker_network.sh`
- `testing/run_tests_from_container.sh`
- `testing/run_tests_from_docker.sh`
- `testing/run_tests_in_container.sh`

**Phase 4 Decision:** Consolidate features, then delete

---

## Rollback (If Needed)

Phase 2 only created a directory - safe to proceed.

If rollback needed:
```bash
rmdir /home/aModels/scripts/dev-tools
# Restore from backup if files were moved accidentally
```

---

## Safety Checklist

- ✅ Backup exists: `/tmp/scripts_backup_20251112.tar.gz`
- ✅ New directory created successfully
- ✅ All 14 source files verified to exist
- ✅ All target directories exist and ready
- ✅ No files modified or deleted yet
- ✅ Rollback available if needed

---

## Expected Results After Phase 3

### Final Distribution Preview

```
scripts/
├── data/           11 files ← No change
├── dev-tools/       5 files ← NEW: Build, download, cleanup utilities
├── lib/             1 file  ← No change
├── quality/         6 files ← +1 performance benchmark
├── signavio/        3 files ← +1 extract script
├── system/          8 files ← +1 start_all_services.sh
└── testing/        10 files ← +6 essential test scripts

Total: 44 organized scripts (from 30)
```

### Directories to Clean (Phase 6)

After moves, these may be empty:
- `/home/aModels/tools/scripts/` (4 files → 0)
- `/home/aModels/tools/helpers/` (1 file → 0)
- `/home/aModels/tools/` (if subdirs empty)

Partially emptied:
- `/home/aModels/testing/` (18 files → 11 redundant scripts remaining)

---

## Status

✅ **Phase 1: COMPLETE** (Backup created)  
✅ **Phase 2: COMPLETE** (Structure ready)  
⏭️ **Phase 3: Ready to execute** (Move 14 files)

**Estimated time for Phase 3:** 5-10 minutes

---

## Ready to Proceed?

### Option 1: Execute Phase 3 Manually
Run the move commands listed above one category at a time.

### Option 2: Use Migration Script
```bash
./scripts/migrate_scripts.sh --execute
```
Will execute Phases 3-6 automatically.

### Option 3: I Can Execute for You
Just say "phase 3" and I'll move the files safely.

---

## Questions?

1. **Will this break anything?**
   - No services broken (only moving scripts, not changing functionality)
   - References in Makefile/docs will need updates (Phase 5)
   - Backup available for full rollback

2. **What about the redundant test scripts?**
   - Staying in `/testing/` for now
   - Will consolidate in Phase 4
   - After testing, delete in Phase 6

3. **Can we test after Phase 3?**
   - Yes! All moved scripts should work from new locations
   - Test with: `scripts/testing/run_all_tests_with_check.sh`
   - Or: `scripts/system/start-system.sh`
