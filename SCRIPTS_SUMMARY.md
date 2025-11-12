# Shell Scripts Summary - aModels Project

## Quick Stats

| Metric | Current | After Reorganization |
|--------|---------|---------------------|
| **Total Scripts** | 71 | 55-60 (consolidated) |
| **Top-Level Directories** | 8+ | 2 (scripts/ + service-specific) |
| **Redundant Scripts** | 10-15 | 0 |
| **Organized Categories** | 5 | 7 |

## Directory Comparison

### Current Structure (Fragmented)
```
/home/aModels/
├── scripts/               ← 30 files (well-organized)
│   ├── data/             (11 scripts)
│   ├── quality/          (5 scripts)
│   ├── signavio/         (2 scripts)
│   ├── system/           (7 scripts)
│   ├── testing/          (4 scripts)
│   └── lib/              (1 script)
│
├── testing/              ← 16 files (REDUNDANT - should be in scripts/)
│   └── run_all_tests*.sh (6 variants!)
│
├── tools/                ← 4 files (scattered utilities)
│   ├── helpers/
│   └── scripts/
│
├── services/             ← 2 misplaced system scripts
│   ├── start_all_services.sh    (should be in scripts/system/)
│   └── telemetry-exporter/create_signavio_extract.sh
│
└── infrastructure/       ← 1 dev utility
    └── docker/brev/sync-testing.sh
```

### Proposed Structure (Consolidated)
```
/home/aModels/
├── scripts/               ← 50-55 files (ALL centralized)
│   ├── data/             (11 scripts) ✓ No change
│   ├── dev-tools/        (6 scripts) ✨ NEW - consolidated utilities
│   ├── lib/              (1 script)  ✓ No change
│   ├── quality/          (6 scripts) + performance_benchmark_runner.sh
│   ├── signavio/         (4 scripts) + create_signavio_extract.sh
│   ├── system/           (8 scripts) + start_all_services.sh
│   └── testing/          (10 scripts) ← Consolidated from /testing/
│
├── services/[service]/    ← Service-specific scripts only
│   ├── agentflow/        (5 scripts) ✓ Stays
│   ├── browser/          (2 scripts) ✓ Stays
│   ├── extract/          (7 scripts) ✓ Stays
│   ├── localai/          (14 scripts) ✓ Stays
│   ├── postgres/         (8 scripts) ✓ Stays
│   └── ...
│
└── data/training/sgmi/    ← SGMI-specific scripts
    └── sgmi-scripts/     (14 scripts) ✓ Stays (domain-specific)
```

## Key Changes

### ✅ Benefits

1. **Single Source of Truth**
   - All general scripts in `/scripts/` with clear categories
   - No more hunting across multiple directories

2. **Eliminated Redundancy**
   - 6 variants of `run_all_tests*.sh` → 1 consolidated script
   - 9 Docker test runners → 2 consolidated scripts
   - **~15 scripts removed**

3. **New Dev-Tools Category**
   - Build utilities
   - Model downloads
   - Cleanup scripts
   - Development helpers

4. **Better Organization**
   - Scripts categorized by function, not by technology
   - Clear separation: system vs service-specific
   - Quality/performance scripts together

5. **Service-Specific Stays Put**
   - LocalAI model management → stays in services/localai/
   - Postgres migrations → stays in services/postgres/
   - Extract ETL → stays in services/extract/
   - **Principle:** Service-specific = stays with service

### 📊 Migration Impact

| Category | Files Moved | Files Consolidated | Net Change |
|----------|-------------|-------------------|------------|
| Dev Tools | 5 → dev-tools/ | - | +5 new category |
| Testing | 16 → testing/ | 10 deleted | -10 redundant |
| System | 1 → system/ | - | +1 |
| Signavio | 1 → signavio/ | - | +1 |
| Quality | 1 → quality/ | - | +1 |
| **Total** | **23 moved** | **10 deleted** | **-13 files** |

## Files to Move

### Priority 1: Testing Scripts (16 files)
**From:** `/testing/`  
**To:** `/scripts/testing/`

**Keep (6 files):**
- ✅ `00_check_services.sh`
- ✅ `bootstrap_training_shell.sh`
- ✅ `run_all_tests.sh` (consolidated)
- ✅ `run_smoke_tests.sh`
- ✅ `setup_test_database.sh`
- ✅ `test_localai_from_container.sh`

**Consolidate & Delete (10 files):**
- ❌ `run_all_tests_final.sh` → merge into run_all_tests.sh
- ❌ `run_all_tests_fixed.sh` → merge into run_all_tests.sh
- ❌ `run_all_tests_with_check.sh` → merge into run_all_tests.sh
- ❌ `run_all_tests_with_step0.sh` → merge into run_all_tests.sh
- ❌ `run_all_tests_working.sh` → merge into run_all_tests.sh
- ❌ `run_tests_docker_network.sh` → merge into run_tests_docker.sh
- ❌ `run_tests_from_container.sh` → merge into run_tests_docker.sh
- ❌ `run_tests_from_docker.sh` → merge into run_tests_docker.sh
- ❌ `run_tests_in_container.sh` → merge into run_tests_docker.sh
- ❌ `run_tests_now.sh` → merge into run_all_tests.sh

**Move to Quality:**
- 📊 `performance_benchmark_runner.sh` → `/scripts/quality/`

### Priority 2: Dev Tools (5 files)
**To:** `/scripts/dev-tools/` (NEW)

- `/tools/helpers/fetch_kaggle_gemma.sh`
- `/tools/scripts/build.sh`
- `/tools/scripts/cleanup_rt_archives.sh`
- `/tools/scripts/run_rt_main_schedule.sh`
- `/infrastructure/docker/brev/sync-testing.sh`

### Priority 3: System Scripts (1 file)
**To:** `/scripts/system/`

- `/services/start_all_services.sh`

### Priority 4: Signavio Scripts (1 file)
**To:** `/scripts/signavio/`

- `/services/telemetry-exporter/create_signavio_extract.sh`

## Migration Commands

### Dry Run (Recommended First)
```bash
cd /home/aModels
./scripts/migrate_scripts.sh --dry-run
```

### Execute Migration
```bash
cd /home/aModels
./scripts/migrate_scripts.sh --execute
```

### Rollback (if needed)
```bash
cd /home/aModels
./scripts/migrate_scripts.sh --rollback
```

## Post-Migration Checklist

- [ ] Run migration script in dry-run mode
- [ ] Review dry-run output
- [ ] Execute migration
- [ ] Update `Makefile.services` (script paths)
- [ ] Update `scripts/README.md` (new structure)
- [ ] Update `docs/SERVICES_STARTUP.md` (references)
- [ ] Update `QUICKSTART.md` (commands)
- [ ] Test system startup: `make -f Makefile.services quick-start`
- [ ] Test health checks: `make -f Makefile.services health`
- [ ] Test script execution from new locations
- [ ] Review and delete redundant scripts
- [ ] Remove empty directories
- [ ] Commit changes with clear message

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Broken references | Medium | High | Automated backup, grep search, dry-run |
| Lost scripts | Low | High | Full backup before migration |
| Service disruption | Low | Medium | Incremental migration, testing |
| Documentation lag | High | Low | Update docs immediately after |

## Documentation to Update

1. **Primary:**
   - `/scripts/README.md` - Script organization guide
   - `/scripts/REORGANIZATION.md` - Migration history
   - `Makefile.services` - All script paths

2. **Secondary:**
   - `docs/SERVICES_STARTUP.md` - System startup procedures
   - `QUICKSTART.md` - Quick reference commands
   - Any CI/CD configuration files

3. **Optional:**
   - Add deprecation notices to old locations
   - Create symlinks for transition period (if needed)

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Review** | 1 hour | Audit redundant scripts, confirm plan |
| **Backup** | 5 min | Create full backup |
| **Migrate** | 30 min | Run migration script, move files |
| **Update Refs** | 1 hour | Update Makefile, docs, scripts |
| **Test** | 30 min | Verify all services still work |
| **Cleanup** | 15 min | Delete redundant scripts, empty dirs |
| **Total** | ~3.5 hours | End-to-end reorganization |

## Questions to Answer

1. **Which test scripts are truly different?**
   - Compare: `run_all_tests.sh` vs `run_all_tests_final.sh` vs `run_all_tests_fixed.sh`
   - Keep the best one, delete duplicates

2. **Are there hard-coded paths in scripts?**
   - Search for: `grep -r "/testing/" scripts/`
   - Search for: `grep -r "/tools/" scripts/`

3. **Do any CI/CD systems reference these paths?**
   - Check GitHub Actions
   - Check GitLab CI
   - Check Jenkins jobs

## Success Criteria

✅ All scripts categorized properly  
✅ No redundant/duplicate scripts  
✅ All references updated  
✅ All services start successfully  
✅ All tests pass  
✅ Documentation reflects new structure  
✅ Old directories cleaned up  

## Need Help?

- See: `SCRIPTS_REORGANIZATION_PLAN.md` for detailed plan
- Run: `./scripts/migrate_scripts.sh --help`
- Review: Migration report after execution
