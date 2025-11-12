# Redundant Test Scripts Analysis

## Summary

**Total Redundant Scripts:** 11  
**Recommendation:** Keep 2-3, delete 8-9

## Detailed Comparison

### Category A: Main Test Runners (6 scripts)

| Script | Lines | Purpose | Set Flags | Keep? |
|--------|-------|---------|-----------|-------|
| `run_all_tests.sh` | 155 | Docker network URLs | `set +e` | ✅ **KEEP** |
| `run_all_tests_final.sh` | 137 | From Docker with mounts | `set +e` | ❌ Merge into main |
| `run_all_tests_fixed.sh` | 164 | Host + Docker scenarios | `set -e` | ❌ Merge into main |
| `run_all_tests_with_check.sh` | 36 | Step 0 check first | `set -e` | ⚠️ **MAYBE** (wrapper) |
| `run_all_tests_with_step0.sh` | 45 | Step 0 then tests | `set +e` | ❌ Duplicate of above |
| `run_all_tests_working.sh` | 141 | Error handling + checks | `set -e` | ❌ Merge into main |

**Analysis:**
- **Primary difference:** Error handling (`set -e` vs `set +e`) and service URL configuration
- **Best approach:** Keep `run_all_tests.sh` as the canonical version
- **Optional:** Keep `run_all_tests_with_check.sh` as a wrapper that calls 00_check_services.sh first

**Recommendation:**
```bash
# KEEP (1-2 scripts)
run_all_tests.sh              # Main test runner
run_all_tests_with_check.sh   # Optional: wrapper with pre-check

# DELETE (4 scripts) - merge features into main
run_all_tests_final.sh
run_all_tests_fixed.sh
run_all_tests_with_step0.sh
run_all_tests_working.sh
```

### Category B: Docker Test Runners (5 scripts)

| Script | Lines | Purpose | Context | Keep? |
|--------|-------|---------|---------|-------|
| `run_tests_docker.sh` | 30 | Docker network context | Generic | ✅ **KEEP** (simple) |
| `run_tests_docker_network.sh` | 40 | Docker network URLs | Generic | ❌ Duplicate |
| `run_tests_from_container.sh` | 153 | From DeepAgents container | Specific | ❌ Too specific |
| `run_tests_from_docker.sh` | 71 | From training-shell | Specific | ❌ Too specific |
| `run_tests_in_container.sh` | 127 | Temporary container | Specific | ❌ Too specific |

**Analysis:**
- **Primary difference:** Which specific container to run from
- **Best approach:** Keep one simple Docker runner, document container-specific usage

**Recommendation:**
```bash
# KEEP (1 script)
run_tests_docker.sh           # Simple Docker network test runner

# DELETE (4 scripts) - consolidate into main docker script
run_tests_docker_network.sh
run_tests_from_container.sh
run_tests_from_docker.sh
run_tests_in_container.sh
```

### Category C: One-Off Runner (1 script)

| Script | Lines | Purpose | Keep? |
|--------|-------|---------|-------|
| `run_tests_now.sh` | 58 | Environment config | ❌ DELETE |

**Analysis:** Likely an old/temporary script with no unique value.

## Consolidation Strategy

### Step 1: Create Canonical Scripts

#### Option A: Minimal (2 scripts)
```
scripts/testing/
├── run_all_tests.sh          # Main test runner (consolidates all features)
└── run_tests_docker.sh       # Docker-specific runner
```

#### Option B: With Pre-Check (3 scripts)
```
scripts/testing/
├── run_all_tests.sh          # Main test runner
├── run_all_tests_with_check.sh  # Wrapper: checks services first
└── run_tests_docker.sh       # Docker-specific runner
```

### Step 2: Feature Consolidation Matrix

| Feature | Source Script(s) | Add to Target |
|---------|------------------|---------------|
| **Error handling (set +e)** | run_all_tests.sh | ✅ Keep as-is |
| **Docker network URLs** | run_all_tests_fixed.sh | ✅ Add to main |
| **Service mounts** | run_all_tests_final.sh | ✅ Add to main |
| **Pre-service check** | run_all_tests_with_check.sh | ⚠️ Keep as wrapper |
| **Container exec modes** | run_tests_from_*.sh | ❌ Document only |

### Step 3: Enhanced run_all_tests.sh

**Proposed consolidated script should include:**

```bash
#!/bin/bash
# Consolidated test runner for aModels project
# Features:
# - Supports both host and Docker network URLs
# - Proper error handling with collection of all results
# - Service availability check (optional via --check flag)
# - Docker mount support
# - Multiple container execution modes

set +e  # Collect all results

# Parse options
CHECK_SERVICES=false
DOCKER_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check) CHECK_SERVICES=true; shift ;;
        --docker) DOCKER_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Optional service check
if [[ "$CHECK_SERVICES" == "true" ]]; then
    ./testing/00_check_services.sh || {
        echo "Service check failed"
        exit 1
    }
fi

# Configure URLs based on mode
if [[ "$DOCKER_MODE" == "true" ]]; then
    export LOCALAI_URL="http://localai:8080"
    export NEO4J_URL="bolt://neo4j:7687"
    # ... other Docker network URLs
else
    export LOCALAI_URL="http://localhost:8080"
    export NEO4J_URL="bolt://localhost:7687"
    # ... other localhost URLs
fi

# Run all tests
echo "Running all tests..."
# ... test execution logic from best variant
```

### Step 4: Simple run_tests_docker.sh

```bash
#!/bin/bash
# Simple Docker network test runner
# Calls main test runner with Docker mode

set -e
cd "$(dirname "$0")"

echo "Running tests in Docker network mode..."
./run_all_tests.sh --docker "$@"
```

## Migration Commands

### Analyze Differences First
```bash
# Compare main variants
cd /home/aModels/testing
diff -u run_all_tests.sh run_all_tests_final.sh > /tmp/diff_final.txt
diff -u run_all_tests.sh run_all_tests_fixed.sh > /tmp/diff_fixed.txt
diff -u run_all_tests.sh run_all_tests_working.sh > /tmp/diff_working.txt

# Review unique features
less /tmp/diff_*.txt
```

### Extract Best Features
```bash
# Look for unique environment variables
grep -h "export\|URL\|HOST" run_all_tests*.sh | sort -u

# Look for unique error handling
grep -h "set \|trap\|error" run_all_tests*.sh | sort -u

# Look for unique service checks
grep -h "check\|health\|wait" run_all_tests*.sh | sort -u
```

### Backup Before Deletion
```bash
# Create backup of redundant scripts
mkdir -p /tmp/redundant_scripts_backup
cp -p run_all_tests_{final,fixed,working,with_step0}.sh /tmp/redundant_scripts_backup/
cp -p run_tests_{docker_network,from_*,in_container,now}.sh /tmp/redundant_scripts_backup/
```

### Delete Redundant Scripts (After Consolidation)
```bash
cd /home/aModels/testing

# After extracting useful features into run_all_tests.sh:
rm -f run_all_tests_final.sh
rm -f run_all_tests_fixed.sh  
rm -f run_all_tests_working.sh
rm -f run_all_tests_with_step0.sh
rm -f run_tests_docker_network.sh
rm -f run_tests_from_container.sh
rm -f run_tests_from_docker.sh
rm -f run_tests_in_container.sh
rm -f run_tests_now.sh

# Result: 9 files deleted, features consolidated
```

## Testing After Consolidation

### Test Matrix

| Scenario | Command | Expected Result |
|----------|---------|-----------------|
| **Host mode** | `./run_all_tests.sh` | Tests run with localhost URLs |
| **Docker mode** | `./run_all_tests.sh --docker` | Tests run with Docker URLs |
| **With check** | `./run_all_tests.sh --check` | Service check runs first |
| **Docker wrapper** | `./run_tests_docker.sh` | Calls main with --docker |

### Validation Checklist

- [ ] All tests pass in host mode
- [ ] All tests pass in Docker mode
- [ ] Service check works correctly
- [ ] Error handling collects all results (set +e)
- [ ] Environment variables set correctly
- [ ] Docker URLs resolve properly
- [ ] Localhost URLs resolve properly
- [ ] No broken references in other scripts

## References to Update

After consolidation, search for and update references:

```bash
cd /home/aModels

# Find all references to deleted scripts
grep -r "run_all_tests_final" .
grep -r "run_all_tests_fixed" .
grep -r "run_all_tests_working" .
grep -r "run_tests_from_container" .
grep -r "run_tests_from_docker" .

# Update to use:
# - run_all_tests.sh (main runner)
# - run_all_tests.sh --docker (Docker mode)
# - run_all_tests.sh --check (with service check)
```

## Final Recommendation

### Keep (3 scripts - Option B)
1. **`run_all_tests.sh`** - Consolidated main runner with all features
2. **`run_all_tests_with_check.sh`** - Wrapper for pre-service-check
3. **`run_tests_docker.sh`** - Simple Docker mode wrapper

### Delete (9 scripts)
1. `run_all_tests_final.sh` - Merge Docker features
2. `run_all_tests_fixed.sh` - Merge URL handling
3. `run_all_tests_with_step0.sh` - Superseded by _with_check variant
4. `run_all_tests_working.sh` - Merge error handling
5. `run_tests_docker_network.sh` - Use --docker flag instead
6. `run_tests_from_container.sh` - Too specific, document instead
7. `run_tests_from_docker.sh` - Too specific, document instead
8. `run_tests_in_container.sh` - Too specific, document instead
9. `run_tests_now.sh` - No unique value

### Result
- **Before:** 11 redundant scripts, confusion about which to use
- **After:** 3 clear scripts with well-defined purposes
- **Reduction:** 73% fewer test runner scripts
- **Benefit:** Clear documentation, consistent behavior, easier maintenance

## Next Steps

1. Review diffs between scripts
2. Consolidate unique features into `run_all_tests.sh`
3. Test consolidated script thoroughly
4. Update documentation
5. Delete redundant scripts
6. Update any references in CI/CD or docs
