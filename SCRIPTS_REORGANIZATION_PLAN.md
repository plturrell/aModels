# Shell Scripts Reorganization Plan

## Executive Summary

Current state: **71 shell scripts** scattered across multiple directories with significant redundancy.

**Primary Issues:**
1. **Duplicate testing directory** - `/testing/` has 16 scripts that should be in `/scripts/testing/`
2. **Inconsistent locations** - Service-level scripts mixed with system scripts
3. **Missing dev-tools category** - Helper/utility scripts not properly organized
4. **Orphaned scripts** - Several standalone scripts in `/services/` root

## Current Distribution

### ✅ Well-Organized (30 files)
- `/scripts/data/` - 11 SGMI data management scripts
- `/scripts/quality/` - 5 quality check scripts  
- `/scripts/system/` - 7 system startup & Docker scripts
- `/scripts/testing/` - 4 test orchestration scripts
- `/scripts/signavio/` - 2 Signavio test data generators
- `/scripts/lib/` - 1 shared utility library

### ❌ Needs Reorganization (41+ files)

#### **Priority 1: Consolidate Testing Scripts**
**Problem:** 16 duplicate/redundant test scripts in `/testing/` (root level)

| Current Location | Purpose | Action |
|------------------|---------|--------|
| `/testing/00_check_services.sh` | Service health check | **MOVE** to `/scripts/testing/` |
| `/testing/run_all_tests*.sh` (6 variants) | Test runners | **CONSOLIDATE** to 1-2 scripts |
| `/testing/run_tests_*.sh` (9 variants) | Docker test runners | **CONSOLIDATE** & move |
| `/testing/performance_benchmark_runner.sh` | Benchmarks | **MOVE** to `/scripts/quality/` |
| `/testing/setup_test_database.sh` | Setup | **MOVE** to `/scripts/testing/` |
| `/testing/bootstrap_training_shell.sh` | Bootstrap | **MOVE** to `/scripts/testing/` |

**Recommendation:** 
- Keep only 3-4 essential test scripts
- Delete redundant variants (run_all_tests_final.sh, run_all_tests_fixed.sh, etc.)
- Move performance benchmarking to quality category

#### **Priority 2: Create Dev-Tools Category**
**Problem:** Development utilities scattered across `/tools/` and `/infrastructure/`

| Current Location | Purpose | New Location |
|------------------|---------|--------------|
| `/tools/helpers/fetch_kaggle_gemma.sh` | Model download | `/scripts/dev-tools/` |
| `/tools/scripts/build.sh` | Build utility | `/scripts/dev-tools/` |
| `/tools/scripts/cleanup_rt_archives.sh` | Cleanup | `/scripts/dev-tools/` |
| `/tools/scripts/run_rt_main_schedule.sh` | Scheduler | `/scripts/dev-tools/` |
| `/infrastructure/docker/brev/sync-testing.sh` | Sync utility | `/scripts/dev-tools/` |

#### **Priority 3: Move Misplaced Service Scripts**
**Problem:** System-wide scripts in `/services/` root

| Current Location | Purpose | New Location |
|------------------|---------|--------------|
| `/services/start_all_services.sh` | System startup | `/scripts/system/` |
| `/services/telemetry-exporter/create_signavio_extract.sh` | Signavio data | `/scripts/signavio/` |

#### **Priority 4: Service-Specific Scripts (Keep as-is)**
These are correctly located within their service directories:

- `/services/agentflow/` - 5 scripts (frontend & flow management)
- `/services/browser/` - 2 scripts (backend & GitHub sync)
- `/services/dms/` - 1 test script
- `/services/extract/scripts/` - 7 ETL pipeline scripts
- `/services/gateway/` - 1 startup script
- `/services/graph/` - 1 dependency check
- `/services/localai/scripts/` - 14 model & service management scripts
- `/services/postgres/scripts/` - 8 database management scripts
- `/services/regulatory/` - 1 audit server launcher
- `/services/orchestration/dashboard/` - 1 test setup

**Rationale:** These are service-specific and should remain with their services.

## Recommended Final Structure

```
/home/aModels/scripts/
├── data/               # 11 files (current)
├── dev-tools/          # 6 files (NEW - consolidated utilities)
│   ├── build.sh
│   ├── cleanup_rt_archives.sh
│   ├── fetch_kaggle_gemma.sh
│   ├── run_rt_main_schedule.sh
│   ├── sync-testing.sh
│   └── visualize_data.sh
├── lib/                # 1 file (current)
├── quality/            # 6 files (add performance_benchmark_runner.sh)
├── signavio/           # 4 files (add create_signavio_extract.sh)
├── system/             # 8 files (add start_all_services.sh)
└── testing/            # 8-10 files (consolidated from /testing/)
    ├── 00_check_services.sh
    ├── bootstrap_training_shell.sh
    ├── run_all_tests.sh (CONSOLIDATED)
    ├── run_smoke_tests.sh
    ├── run_tests_docker.sh (CONSOLIDATED)
    ├── setup_test_database.sh
    ├── test_all_services.sh
    ├── test_improvements.sh
    ├── test_localai_from_container.sh
    └── test_sgmi_end_to_end.sh

/data/training/sgmi/sgmi-scripts/  # 14 files (keep as-is - SGMI specific)

/services/[service]/               # Service-specific scripts remain
```

## Migration Steps

### Phase 1: Backup
```bash
tar -czf /tmp/scripts_backup_$(date +%Y%m%d).tar.gz \
  /home/aModels/scripts \
  /home/aModels/testing \
  /home/aModels/tools
```

### Phase 2: Create New Structure
```bash
mkdir -p /home/aModels/scripts/dev-tools
```

### Phase 3: Move Files
```bash
# Move dev tools
mv /home/aModels/tools/helpers/fetch_kaggle_gemma.sh /home/aModels/scripts/dev-tools/
mv /home/aModels/tools/scripts/*.sh /home/aModels/scripts/dev-tools/
mv /home/aModels/infrastructure/docker/brev/sync-testing.sh /home/aModels/scripts/dev-tools/

# Move service scripts to system
mv /home/aModels/services/start_all_services.sh /home/aModels/scripts/system/

# Move signavio script
mv /home/aModels/services/telemetry-exporter/create_signavio_extract.sh /home/aModels/scripts/signavio/

# Move testing scripts (essential ones)
mv /home/aModels/testing/00_check_services.sh /home/aModels/scripts/testing/
mv /home/aModels/testing/bootstrap_training_shell.sh /home/aModels/scripts/testing/
mv /home/aModels/testing/setup_test_database.sh /home/aModels/scripts/testing/
mv /home/aModels/testing/run_smoke_tests.sh /home/aModels/scripts/testing/
mv /home/aModels/testing/test_localai_from_container.sh /home/aModels/scripts/testing/

# Move performance to quality
mv /home/aModels/testing/performance_benchmark_runner.sh /home/aModels/scripts/quality/
```

### Phase 4: Consolidate Redundant Scripts
Create `/home/aModels/scripts/testing/run_all_tests.sh` (consolidated) and delete:
- `run_all_tests_final.sh`
- `run_all_tests_fixed.sh`
- `run_all_tests_with_check.sh`
- `run_all_tests_with_step0.sh`
- `run_all_tests_working.sh`

Create `/home/aModels/scripts/testing/run_tests_docker.sh` (consolidated) and delete:
- `run_tests_docker_network.sh`
- `run_tests_from_container.sh`
- `run_tests_from_docker.sh`
- `run_tests_in_container.sh`
- `run_tests_now.sh`

### Phase 5: Update References
Search and update all references to moved scripts:
```bash
grep -r "testing/" /home/aModels/scripts/ /home/aModels/Makefile* /home/aModels/docs/
grep -r "tools/scripts/" /home/aModels/scripts/ /home/aModels/Makefile* /home/aModels/docs/
grep -r "start_all_services.sh" /home/aModels/scripts/ /home/aModels/Makefile* /home/aModels/docs/
```

### Phase 6: Cleanup
```bash
# Remove old directories if empty
rmdir /home/aModels/testing 2>/dev/null || echo "Directory not empty, review contents"
rmdir /home/aModels/tools/scripts 2>/dev/null || echo "Directory not empty, review contents"
rmdir /home/aModels/tools/helpers 2>/dev/null || echo "Directory not empty, review contents"
```

## Expected Outcome

**Before:** 71 scripts across 8+ directories
**After:** 55-60 scripts in organized structure

**Benefits:**
1. ✅ Single source of truth for scripts (`/scripts/`)
2. ✅ Clear categorization (data, quality, system, testing, dev-tools, signavio)
3. ✅ Reduced redundancy (eliminate 10-15 duplicate test scripts)
4. ✅ Service-specific scripts remain with services
5. ✅ Easier maintenance and discovery

## Files to Review for Deletion

### Redundant Test Scripts (11 candidates)
- `/testing/run_all_tests_final.sh` - duplicate of run_all_tests.sh?
- `/testing/run_all_tests_fixed.sh` - duplicate of run_all_tests.sh?
- `/testing/run_all_tests_with_check.sh` - can consolidate
- `/testing/run_all_tests_with_step0.sh` - can consolidate
- `/testing/run_all_tests_working.sh` - duplicate of run_all_tests.sh?
- `/testing/run_tests_docker_network.sh` - can consolidate
- `/testing/run_tests_from_container.sh` - duplicate
- `/testing/run_tests_from_docker.sh` - duplicate
- `/testing/run_tests_in_container.sh` - duplicate
- `/testing/run_tests_now.sh` - duplicate of run_all_tests.sh?

**Action Required:** Review each script to determine which are truly needed.

## Dependencies to Update

After reorganization, update these files:
1. `/home/aModels/Makefile.services` - Update all script paths
2. `/home/aModels/scripts/README.md` - Update documentation
3. `/home/aModels/docs/SERVICES_STARTUP.md` - Update references
4. `/home/aModels/QUICKSTART.md` - Update quick start commands
5. Any CI/CD pipelines or automation that references old paths

## Risk Mitigation

1. **Backup created** before any changes
2. **Incremental migration** - move category by category
3. **Test after each phase** - ensure nothing breaks
4. **Keep old paths** temporarily with deprecation notices
5. **Document all changes** in CHANGELOG

## Next Steps

1. **Review this plan** - Confirm approach
2. **Audit redundant scripts** - Determine which test scripts to keep
3. **Execute Phase 1** - Create backup
4. **Execute Phases 2-3** - Move files incrementally
5. **Test thoroughly** - Run `make -f Makefile.services health`
6. **Update documentation** - Reflect new structure
7. **Remove old directories** - After validation period
