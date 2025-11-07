# Fix Priorities for Disabled Code

Prioritized list of disabled code items, categorized by urgency and importance.

## Priority Levels

- **HIGH:** Code that should be working but is broken, blocking functionality
- **MEDIUM:** Code that may be needed in future, or has workarounds available
- **LOW:** Code that's intentionally disabled or obsolete, low impact
- **NO ACTION:** Code that's correctly disabled for valid reasons

---

## HIGH PRIORITY

### 1. Arrow Version Conflicts in Test Files
**Priority:** HIGH  
**Impact:** Test coverage gaps, potential runtime issues

**Files:**
- `services/agentflow/internal/catalog/flightcatalog/flightcatalog_test.go`
- `services/graph/internal/catalog/flightcatalog/flightcatalog_test.go`
- `services/graph/cmd/graph-server/catalog_test.go`

**Issue:** Arrow v16 vs v18 conflicts prevent tests from running

**Why High Priority:**
- Tests are critical for ensuring catalog functionality works
- Version conflicts may indicate runtime issues
- Missing test coverage for important features

**Action:** Resolve Arrow dependency version conflicts or update tests to use consistent version

---

## MEDIUM PRIORITY

### 2. Missing `ai_benchmarks` Package Dependencies
**Priority:** MEDIUM  
**Impact:** Benchmarking and tuning tools unavailable

**Files:** 18 files (mappers, commands, packages)

**Why Medium Priority:**
- Benchmarking tools are useful but not critical for core functionality
- May be needed for performance optimization work
- Could be replaced with alternative benchmarking solutions

**Options:**
1. Add `ai_benchmarks` as dependency if package exists
2. Remove files if functionality is no longer needed
3. Create stub implementations if not critical

**Recommendation:** Determine if benchmarking is actively used. If not, remove files. If yes, add dependency or find alternative.

---

### 3. Missing AgentSDK Package Dependencies
**Priority:** MEDIUM  
**Impact:** Flight catalog and enhanced inference features unavailable

**Files:**
- `pkg/catalog/flightcatalog/flightcatalog.go`
- `pkg/localai/enhanced_inference.go`
- `tools/cmd/benchmark-server/main.go`

**Why Medium Priority:**
- Flight catalog may be needed for agent orchestration
- Enhanced inference could be useful feature
- May be part of planned architecture

**Action:** Determine if AgentSDK package exists and should be added, or if these features are obsolete

---

### 4. Stress Testing Functions (Commented Code)
**Priority:** MEDIUM  
**Impact:** Missing stress testing capabilities

**File:** `infrastructure/third_party/orchestration/stress_testing.go`

**Why Medium Priority:**
- Stress testing is valuable for production readiness
- Large amount of commented code (~900 lines) makes file hard to maintain
- Missing dependencies may indicate architectural gaps

**Options:**
1. Add missing dependencies (privacy, storage, agents packages)
2. Remove commented code if stress testing not needed
3. Refactor to use available dependencies

**Recommendation:** If stress testing is needed, add dependencies. Otherwise, remove commented code to clean up file.

---

### 5. Missing HANA Package Dependencies
**Priority:** MEDIUM  
**Impact:** HANA vectorstore, memory, and tools unavailable

**Files:**
- `infrastructure/third_party/orchestration/vectorstores/hana/hana_vectorstore.go`
- `infrastructure/third_party/orchestration/memory/hana/hana_memory.go`
- `infrastructure/third_party/orchestration/tools/hana/hana_tools.go`
- `pkg/sap/hana_client.go`

**Why Medium Priority:**
- HANA integration may be required for specific deployments
- If not using HANA, these files are unnecessary
- Could be important for enterprise customers

**Action:** Determine if HANA integration is needed. If yes, add dependencies. If no, remove files.

---

## LOW PRIORITY

### 6. Missing Cloud SQL Package Dependencies
**Priority:** LOW  
**Impact:** Cloud SQL utilities unavailable

**Files:**
- `infrastructure/third_party/orchestration/util/cloudsqlutil/engine.go`
- `infrastructure/third_party/orchestration/util/cloudsqlutil/options.go`

**Why Low Priority:**
- Cloud SQL is specific to Google Cloud deployments
- Not needed for all users
- Can be added when needed

**Action:** Add dependency only if Cloud SQL integration is required

---

### 7. Performance Profiler Disabled Functions
**Priority:** LOW  
**Impact:** Some profiling capabilities unavailable

**File:** `infrastructure/third_party/orchestration/performance_profiler.go`

**Why Low Priority:**
- Profiling is useful but not critical
- Other profiling tools may be available
- Missing dependencies indicate optional features

**Action:** Add dependencies if profiling is needed, otherwise leave disabled

---

### 8. Package Name Conflict
**Priority:** LOW  
**Impact:** Testing service main file disabled

**File:** `services/testing/main.go`

**Why Low Priority:**
- Conflicts with standard library `testing` package
- Can be resolved by renaming package or moving file
- Low impact on functionality

**Action:** Rename package or move to different directory

---

## NO ACTION REQUIRED

### 9. Lint-Ignored Functions (Reserved for Future)
**Priority:** NO ACTION  
**Status:** Correctly disabled

**File:** `infrastructure/third_party/orchestration/internal/devtools/lint/lint.go`

**Functions:**
- `checkMissingReplaceDirectives()` - Reserved for development workflows
- `fixAddReplaceDirective()` - Reserved for development workflows

**Why No Action:**
- Intentionally disabled for future use
- Well-documented with comments
- Not needed in current workflow

**Action:** None - these are correctly reserved for future use

---

### 10. Intentionally Ignored Test Files
**Priority:** NO ACTION  
**Status:** Correctly disabled

**Files:**
- `infrastructure/third_party/go-arrow/arrow/gen-flatbuffers.go` - Build tool
- `infrastructure/third_party/orchestration/tools/perplexity/perplexity_test.go` - Test file
- `infrastructure/third_party/orchestration/memory/token_buffer_test.go` - Test file
- `infrastructure/third_party/orchestration/llms/compliance/example_test.go` - Example test
- `infrastructure/third_party/orchestration/chains/llm_test.go` - Test file
- `infrastructure/third_party/orchestration/chains/constitution/constitutional_test.go` - Test file

**Why No Action:**
- These are intentionally ignored for valid reasons (build tools, example tests)
- No functionality impact

**Action:** None

---

## Summary by Priority

| Priority | Count | Examples |
|----------|-------|----------|
| **HIGH** | 3 files | Arrow version conflicts in tests |
| **MEDIUM** | 25+ files | ai_benchmarks, AgentSDK, stress testing, HANA |
| **LOW** | 7 files | Cloud SQL, performance profiler, package conflicts |
| **NO ACTION** | 8 files | Lint-ignored (future use), intentionally ignored tests |

---

## Recommended Action Order

1. **First:** Resolve Arrow version conflicts (HIGH priority, affects test coverage)
2. **Second:** Evaluate and decide on ai_benchmarks dependencies (MEDIUM priority, affects 18 files)
3. **Third:** Clean up stress_testing.go (MEDIUM priority, large commented blocks)
4. **Fourth:** Evaluate AgentSDK dependencies (MEDIUM priority, affects catalog features)
5. **Fifth:** Evaluate HANA dependencies (MEDIUM priority, affects enterprise features)
6. **Last:** Address LOW priority items as needed

---

## Decision Points Needed

Before fixing, determine:

1. **Is `ai_benchmarks` package available?** If yes, add as dependency. If no, remove 18 files.
2. **Is AgentSDK package available?** If yes, add as dependency. If no, remove catalog files.
3. **Is stress testing needed?** If yes, add dependencies. If no, remove commented code.
4. **Is HANA integration needed?** If yes, add dependencies. If no, remove HANA files.
5. **Which Arrow version should be used?** Resolve v16 vs v18 conflict consistently.

