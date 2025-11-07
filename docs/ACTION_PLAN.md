# Action Plan for Disabled Code

Specific steps to fix or remove disabled code, organized by priority.

---

## Phase 1: HIGH PRIORITY - Arrow Version Conflicts

### Goal: Resolve Arrow version conflicts in test files

### Step 1.1: Investigate Arrow Dependencies
```bash
# Check current Arrow dependencies
grep -r "apache/arrow" go.mod go.sum
grep -r "github.com/apache/arrow" **/go.mod
```

**Action Items:**
1. Identify which Arrow version is used in the main codebase
2. Check if v16 and v18 can coexist (check import paths)
3. Determine if tests can be updated to use the main version

### Step 1.2: Update Test Files
**Files to update:**
- `services/agentflow/internal/catalog/flightcatalog/flightcatalog_test.go`
- `services/graph/internal/catalog/flightcatalog/flightcatalog_test.go`
- `services/graph/cmd/graph-server/catalog_test.go`

**Options:**
- **Option A:** Update all tests to use Arrow v18 (if main codebase uses v18)
- **Option B:** Update all tests to use Arrow v16 (if main codebase uses v16)
- **Option C:** Use build tags to support both versions

**Recommended:** Option A or B (pick one version consistently)

### Step 1.3: Remove Build Ignore Tags
Once version conflicts are resolved:
```go
// Remove these lines from each test file:
//go:build ignore
// +build ignore
```

### Step 1.4: Verify Tests Run
```bash
go test ./services/agentflow/internal/catalog/flightcatalog/...
go test ./services/graph/internal/catalog/flightcatalog/...
go test ./services/graph/cmd/graph-server/...
```

**Estimated Time:** 2-4 hours

---

## Phase 2: MEDIUM PRIORITY - Evaluate and Fix Dependencies

### Goal: Determine which dependencies are needed and add or remove accordingly

### Step 2.1: Evaluate `ai_benchmarks` Package

**Investigation:**
```bash
# Check if package exists
go list -m -versions ai_benchmarks 2>&1
# Or search for it in go.mod files
grep -r "ai_benchmarks" **/go.mod
```

**Decision Tree:**
1. **If package exists:**
   - Add to appropriate `go.mod` files
   - Remove `//go:build ignore` from all 18 files
   - Run `go mod tidy`

2. **If package doesn't exist but is needed:**
   - Create the package or find alternative
   - Document decision

3. **If package is not needed:**
   - Delete all 18 files with `ai_benchmarks` dependencies
   - Clean up any references

**Files affected:** 18 files (see DISABLED_CODE_CATALOG.md section 1.1)

**Action:**
```bash
# If removing files:
rm tools/scripts/factory/mappers/{triviaqa,socialiqa,piqa,hellaswag,boolq,arc}_mapper.go
rm tools/cmd/aibench/{tune,main,nash}.go
rm tools/cmd/factory/main.go
rm tools/cmd/calibrate/main.go
rm tools/cmd/benchmark-server/main.go
rm pkg/lnn/{tune,nash}.go
rm pkg/localai/enhanced_inference.go  # Also needs AgentSDK check
rm pkg/models/maths_mcq.go
rm pkg/methods/aggregate.go
rm pkg/preprocess/preprocess.go
rm tools/scripts/factory/connectors/csv_connector.go
```

**Estimated Time:** 2-6 hours (depending on decision)

---

### Step 2.2: Evaluate AgentSDK Package

**Investigation:**
```bash
# Check if package exists
go list -m -versions agenticAiETH_layer4_AgentSDK 2>&1
# Or check if it's in a different location
find . -name "*AgentSDK*" -type d
```

**Decision Tree:**
1. **If package exists:**
   - Add to appropriate `go.mod` files
   - Remove `//go:build ignore` from affected files
   - Run `go mod tidy`

2. **If package doesn't exist:**
   - Determine if these features are needed
   - If not needed, remove files
   - If needed, create package or find alternative

**Files affected:**
- `pkg/catalog/flightcatalog/flightcatalog.go`
- `pkg/localai/enhanced_inference.go` (also needs ai_benchmarks)
- `tools/cmd/benchmark-server/main.go` (also needs ai_benchmarks)
- `infrastructure/third_party/orchestration/catalog/flightcatalog/flightcatalog.go`
- `infrastructure/third_party/orchestration/catalog/flightcatalog/flightcatalog_test.go`

**Estimated Time:** 2-4 hours

---

### Step 2.3: Clean Up Stress Testing File

**File:** `infrastructure/third_party/orchestration/stress_testing.go`

**Option A: Remove Commented Code (Recommended if not needed)**
```bash
# Create a script to remove commented blocks
# Or manually remove all /* ... */ blocks
```

**Steps:**
1. Review each disabled function
2. If not needed, remove the entire commented block
3. Keep function signatures with error returns if they're part of an interface
4. Update documentation

**Option B: Add Dependencies (If stress testing is needed)**
1. Add missing packages:
   - `agenticAiETH_layer4_HANA/pkg/privacy`
   - `agenticAiETH_layer4_HANA/pkg/storage`
   - `agenticAiETH_layer1_Blockchain/infrastructure/common`
   - `agenticAiETH_layer1_Blockchain/processes/agents`

2. Uncomment code blocks
3. Remove `DISABLED:` comments
4. Test stress testing functionality

**Recommended:** Option A (remove commented code) unless stress testing is actively needed

**Estimated Time:** 2-3 hours (Option A) or 4-6 hours (Option B)

---

### Step 2.4: Evaluate HANA Dependencies

**Investigation:**
```bash
# Check if HANA packages are available
# Search for HANA-related dependencies
grep -r "hana\|HANA\|sap" **/go.mod
```

**Decision Tree:**
1. **If HANA integration is needed:**
   - Add HANA/SAP packages as dependencies
   - Remove `//go:build ignore` from HANA files
   - Test HANA integration

2. **If HANA integration is not needed:**
   - Remove HANA-related files
   - Clean up references

**Files affected:**
- `infrastructure/third_party/orchestration/vectorstores/hana/hana_vectorstore.go`
- `infrastructure/third_party/orchestration/memory/hana/hana_memory.go`
- `infrastructure/third_party/orchestration/tools/hana/hana_tools.go`
- `pkg/sap/hana_client.go`

**Action (if removing):**
```bash
rm -rf infrastructure/third_party/orchestration/vectorstores/hana/
rm -rf infrastructure/third_party/orchestration/memory/hana/
rm -rf infrastructure/third_party/orchestration/tools/hana/
rm pkg/sap/hana_client.go
```

**Estimated Time:** 1-3 hours

---

## Phase 3: LOW PRIORITY - Optional Fixes

### Step 3.1: Cloud SQL Utilities

**Decision:** Only fix if Cloud SQL integration is needed

**If fixing:**
1. Add `cloud.google.com/go/cloudsqlconn` dependency
2. Remove `//go:build ignore` from:
   - `infrastructure/third_party/orchestration/util/cloudsqlutil/engine.go`
   - `infrastructure/third_party/orchestration/util/cloudsqlutil/options.go`
3. Test Cloud SQL connection functionality

**If removing:**
```bash
rm -rf infrastructure/third_party/orchestration/util/cloudsqlutil/
```

**Estimated Time:** 1-2 hours

---

### Step 3.2: Package Name Conflict

**File:** `services/testing/main.go`

**Fix:**
1. Rename package or move file to avoid conflict with standard library `testing`
2. Options:
   - Rename to `services/testing-service/main.go`
   - Rename package to `testingservice`
   - Move to `services/test-runner/main.go`

**Action:**
```bash
# Option 1: Rename directory
mv services/testing services/testing-service

# Option 2: Rename file
mv services/testing/main.go services/testing-service/main.go
# Update package name in file
```

**Estimated Time:** 30 minutes

---

### Step 3.3: Performance Profiler

**File:** `infrastructure/third_party/orchestration/performance_profiler.go`

**Decision:** Only fix if profiling features are needed

**If fixing:**
1. Add missing dependencies (same as stress testing)
2. Uncomment disabled functions
3. Test profiling functionality

**If not needed:**
- Leave as-is (functions return errors, which is acceptable)

**Estimated Time:** 2-4 hours (if fixing)

---

## Phase 4: Verification and Testing

### Step 4.1: Build Verification
```bash
# After removing build-ignore tags, verify builds work
go build ./...
```

### Step 4.2: Test Verification
```bash
# Run tests for fixed files
go test ./...
```

### Step 4.3: Lint Verification
```bash
# Check for any linting issues
go run ./internal/devtools/lint
```

### Step 4.4: Dependency Check
```bash
# Verify no missing dependencies
go mod verify
go mod tidy
```

**Estimated Time:** 1-2 hours

---

## Quick Reference: File Removal Commands

If decision is made to remove files (not fix):

```bash
# ai_benchmarks files (if removing)
rm tools/scripts/factory/mappers/{triviaqa,socialiqa,piqa,hellaswag,boolq,arc}_mapper.go
rm tools/cmd/aibench/{tune,main,nash}.go
rm tools/cmd/factory/main.go
rm tools/cmd/calibrate/main.go
rm tools/cmd/benchmark-server/main.go
rm pkg/lnn/{tune,nash}.go
rm pkg/models/maths_mcq.go
rm pkg/methods/aggregate.go
rm pkg/preprocess/preprocess.go
rm tools/scripts/factory/connectors/csv_connector.go

# HANA files (if removing)
rm -rf infrastructure/third_party/orchestration/vectorstores/hana/
rm -rf infrastructure/third_party/orchestration/memory/hana/
rm -rf infrastructure/third_party/orchestration/tools/hana/
rm pkg/sap/hana_client.go

# Cloud SQL files (if removing)
rm -rf infrastructure/third_party/orchestration/util/cloudsqlutil/

# AgentSDK files (if removing - after confirming not needed)
rm pkg/catalog/flightcatalog/flightcatalog.go
rm infrastructure/third_party/orchestration/catalog/flightcatalog/flightcatalog.go
rm infrastructure/third_party/orchestration/catalog/flightcatalog/flightcatalog_test.go
```

---

## Decision Checklist

Before starting fixes, answer these questions:

- [ ] Is `ai_benchmarks` package available? (Yes/No/Unknown)
- [ ] Is `agenticAiETH_layer4_AgentSDK` package available? (Yes/No/Unknown)
- [ ] Is stress testing functionality needed? (Yes/No)
- [ ] Is HANA integration needed? (Yes/No)
- [ ] Is Cloud SQL integration needed? (Yes/No)
- [ ] Which Arrow version should be used? (v16/v18/Unknown)
- [ ] Are profiling features needed? (Yes/No)

---

## Estimated Total Time

- **Phase 1 (Arrow conflicts):** 2-4 hours
- **Phase 2 (Dependencies):** 8-16 hours (depending on decisions)
- **Phase 3 (Optional fixes):** 2-6 hours
- **Phase 4 (Verification):** 1-2 hours

**Total:** 13-28 hours (depending on decisions and scope)

---

## Notes

1. **Start with Phase 1** - Arrow conflicts are highest priority and affect test coverage
2. **Make decisions before Phase 2** - Determine which dependencies are actually needed
3. **Remove code if not needed** - Don't keep disabled code "just in case"
4. **Update documentation** - Document decisions and rationale
5. **Test after each phase** - Don't wait until the end to verify changes

