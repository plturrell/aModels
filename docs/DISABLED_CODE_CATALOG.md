# Disabled Code Catalog

Complete inventory of all disabled code in the codebase, organized by category.

## Summary

- **Total Files with `//go:build ignore`:** 38 files
- **Commented-Out Code Blocks:** 1 file (stress_testing.go) with 15+ disabled functions
- **Lint-Ignored Functions:** 2 functions (reserved for future use)
- **Arrow Version Conflicts:** 3 test files

---

## 1. Build-Ignored Packages (`//go:build ignore`)

### 1.1 Missing `ai_benchmarks` Package (18 files)

**Reason:** Dependencies on missing `ai_benchmarks` package

#### Mapper Files (6 files)
- `tools/scripts/factory/mappers/triviaqa_mapper.go` - TriviaQA task mapper
- `tools/scripts/factory/mappers/socialiqa_mapper.go` - SocialIQA task mapper
- `tools/scripts/factory/mappers/piqa_mapper.go` - PIQA task mapper
- `tools/scripts/factory/mappers/hellaswag_mapper.go` - HellaSwag task mapper
- `tools/scripts/factory/mappers/boolq_mapper.go` - BoolQ task mapper
- `tools/scripts/factory/mappers/arc_mapper.go` - ARC task mapper

#### Command Files (6 files)
- `tools/cmd/aibench/tune.go` - Hyperparameter tuning command
- `tools/cmd/aibench/main.go` - Main aibench command
- `tools/cmd/aibench/nash.go` - Nash equilibrium optimization
- `tools/cmd/factory/main.go` - Data factory pipeline
- `tools/cmd/calibrate/main.go` - Calibration command
- `tools/cmd/benchmark-server/main.go` - Benchmark server

#### Package Files (6 files)
- `pkg/lnn/tune.go` - LNN-based tuning
- `pkg/lnn/nash.go` - Nash equilibrium in LNN
- `pkg/localai/enhanced_inference.go` - Enhanced inference (also needs AgentSDK)
- `pkg/models/maths_mcq.go` - Math MCQ models
- `pkg/methods/aggregate.go` - Aggregation methods
- `pkg/preprocess/preprocess.go` - Preprocessing utilities

#### Connector Files (1 file)
- `tools/scripts/factory/connectors/csv_connector.go` - CSV data connector

### 1.2 Missing AgentSDK Package (3 files)

**Reason:** Dependencies on missing `agenticAiETH_layer4_AgentSDK` package

- `pkg/catalog/flightcatalog/flightcatalog.go` - Flight catalog implementation
- `pkg/localai/enhanced_inference.go` - Enhanced inference (also needs ai_benchmarks)
- `tools/cmd/benchmark-server/main.go` - Benchmark server (also needs ai_benchmarks)

### 1.3 Arrow Version Conflicts (3 test files)

**Reason:** Arrow version conflicts (v16 vs v18) and/or missing AgentSDK

- `services/agentflow/internal/catalog/flightcatalog/flightcatalog_test.go` - Arrow v16 + missing AgentSDK
- `services/graph/internal/catalog/flightcatalog/flightcatalog_test.go` - Arrow v16 conflict
- `services/graph/cmd/graph-server/catalog_test.go` - Arrow v16 conflict

### 1.4 Missing HANA Packages (4 files)

**Reason:** Dependencies on missing HANA/SAP packages

- `infrastructure/third_party/orchestration/vectorstores/hana/hana_vectorstore.go`
- `infrastructure/third_party/orchestration/memory/hana/hana_memory.go`
- `infrastructure/third_party/orchestration/tools/hana/hana_tools.go`
- `pkg/sap/hana_client.go` - Also needs agenticAiETH_layer4 package

### 1.5 Missing Cloud SQL Package (2 files)

**Reason:** Dependencies on missing `cloud.google.com/go/cloudsqlconn` package

- `infrastructure/third_party/orchestration/util/cloudsqlutil/engine.go`
- `infrastructure/third_party/orchestration/util/cloudsqlutil/options.go` - Depends on disabled engine.go

### 1.6 Package Name Conflicts (1 file)

**Reason:** Conflicts with standard library `testing` package

- `services/testing/main.go` - Conflicts with package `testing` in same directory

### 1.7 Other Build-Ignored Files (6 files)

**Reason:** Various reasons

- `infrastructure/third_party/orchestration/catalog/flightcatalog/flightcatalog.go` - Missing AgentSDK + Arrow conflicts
- `infrastructure/third_party/orchestration/catalog/flightcatalog/flightcatalog_test.go` - Missing AgentSDK + Arrow conflicts
- `infrastructure/third_party/go-arrow/arrow/gen-flatbuffers.go` - Build tool, intentionally ignored
- `infrastructure/third_party/orchestration/tools/perplexity/perplexity_test.go` - Test file, intentionally ignored
- `infrastructure/third_party/orchestration/memory/token_buffer_test.go` - Test file, intentionally ignored
- `infrastructure/third_party/orchestration/llms/compliance/example_test.go` - Example test, intentionally ignored
- `infrastructure/third_party/orchestration/chains/llm_test.go` - Test file, intentionally ignored
- `infrastructure/third_party/orchestration/chains/constitution/constitutional_test.go` - Test file, intentionally ignored

---

## 2. Commented-Out Code Blocks

### 2.1 Stress Testing Functions (`stress_testing.go`)

**File:** `infrastructure/third_party/orchestration/stress_testing.go`

**Reason:** Missing dependencies (privacy, agents, storage packages)

#### Disabled Functions:

1. **`RunAllStressTests()`** (lines 73-121)
   - Main entry point for all stress tests
   - Missing: privacy, agents, storage packages

2. **`runPrivacyStressTest()`** (lines 124-173)
   - Privacy operations stress test
   - Missing: `privacy` package

3. **`privacyWorker()`** (lines 175-244)
   - Worker implementation for privacy operations
   - Missing: `privacyManager` from privacy package

4. **`runAgentStressTest()`** (lines 246-280)
   - Agent operations stress test
   - Missing: `agents` and `common` packages

5. **`agentWorker()`** (lines 282-347)
   - Worker implementation for agent operations
   - Missing: `searchOps` from agents package

6. **`runVectorStressTest()`** (lines 349-384)
   - Vector operations stress test
   - Missing: `storage` package

7. **`vectorWorker()`** (lines 386-445)
   - Worker implementation for vector operations
   - Missing: `vectorStore` from storage package

8. **`runGraphStressTest()`** (lines 447-481)
   - Graph operations stress test
   - Missing: `storage` package

9. **`graphWorker()`** (lines 483-544)
   - Worker implementation for graph operations
   - Missing: `graphStore` from storage package

10. **`runDatabaseStressTest()`** (lines 546-581)
    - Database operations stress test
    - Missing: `storage` package

11. **`databaseWorker()`** (lines 583-644)
    - Worker implementation for database operations
    - Missing: `relationalStore` from storage package

12. **`runMixedWorkloadStressTest()`** (lines 646-680)
    - Mixed workload stress test
    - Missing: multiple packages (privacy, agents, storage)

13. **`mixedWorkloadWorker()`** (lines 682-744)
    - Worker implementation for mixed workloads
    - Missing: multiple packages

14. **`runConcurrentStressTest()`** (lines 746-786)
    - High concurrency stress test
    - Missing: multiple packages

15. **`concurrentWorker()`** (lines 788-842)
    - Worker implementation for concurrent operations
    - Missing: `privacy` package

16. **`runMemoryPressureStressTest()`** (lines 844-878)
    - Memory pressure stress test
    - Missing: multiple packages

17. **`memoryPressureWorker()`** (lines 880-937)
    - Worker implementation for memory pressure tests
    - Missing: `storage` package

18. **`runConnectionPoolStressTest()`** (lines 939-974)
    - Connection pool stress test
    - Missing: multiple packages

19. **`connectionPoolWorker()`** (lines 976-1027)
    - Worker implementation for connection pool tests
    - Missing: `storage` package

20. **`runRateLimitingStressTest()`** (lines 1029-1063)
    - Rate limiting stress test
    - Missing: multiple packages

21. **`rateLimitingWorker()`** (lines 1065-1119)
    - Worker implementation for rate limiting tests
    - Missing: `privacy` package

22. **`initializeTestEnvironment()`** (lines 1124-1145)
    - Test environment initialization
    - Missing: `privacy` package

**Total Commented Lines:** ~900+ lines of commented-out code

---

## 3. Lint-Ignored Functions (Reserved for Future)

### 3.1 Development Workflow Functions (`lint.go`)

**File:** `infrastructure/third_party/orchestration/internal/devtools/lint/lint.go`

**Reason:** Intentionally reserved for future development workflows, not enabled in standard mode

1. **`checkMissingReplaceDirectives()`** (line 327)
   - Checks if go.mod files have appropriate replace directives for local development
   - Marked with: `//lint:ignore U1000 Reserved for future use in development workflows (not enabled in standard mode)`
   - Purpose: Enable replace directives for local development without publishing

2. **`fixAddReplaceDirective()`** (line 435)
   - Adds replace directives to go.mod files
   - Marked with: `//lint:ignore U1000 Reserved for future use with checkMissingReplaceDirectives`
   - Purpose: Helper function for development workflow

**Status:** Correctly disabled - these are intentionally reserved for future use

---

## 4. Performance Profiler Disabled Functions

**File:** `infrastructure/third_party/orchestration/performance_profiler.go`

**Reason:** Missing dependencies (privacy, storage, agents packages)

1. **Line 270:** Privacy profiling function - Missing `privacy` package
2. **Line 355:** Vector profiling function - Missing `storage` package
3. **Line 410:** Graph profiling function - Missing `storage` package
4. **Line 478:** Agent profiling function - Missing `agents` and `common` packages
5. **Line 533:** Database profiling function - Missing `storage` package

---

## 5. Other Disabled Code

### 5.1 Chain Runner (`cmd/chainrunner/main.go`)

**File:** `infrastructure/third_party/orchestration/cmd/chainrunner/main.go`

**Disabled:**
- Line 13: Import of `catalogprompt` package (missing AgentSDK)
- Line 78: Context text generation using catalogprompt

---

## Dependencies Summary

### Missing External Packages:
1. **`ai_benchmarks`** - Used by 18 files (mappers, commands, packages)
2. **`agenticAiETH_layer4_AgentSDK`** - Used by 3+ files (catalog, enhanced inference)
3. **`agenticAiETH_layer4_HANA/pkg/privacy`** - Used by stress testing and performance profiler
4. **`agenticAiETH_layer4_HANA/pkg/storage`** - Used by stress testing and performance profiler
5. **`agenticAiETH_layer1_Blockchain/infrastructure/common`** - Used by stress testing
6. **`agenticAiETH_layer1_Blockchain/processes/agents`** - Used by stress testing
7. **`cloud.google.com/go/cloudsqlconn`** - Used by Cloud SQL utilities
8. **HANA/SAP packages** - Used by HANA vectorstore, memory, and tools

### Version Conflicts:
- **Apache Arrow:** v16 vs v18 conflicts in 3 test files

---

## File Count by Category

| Category | Count |
|----------|-------|
| Build-ignored (ai_benchmarks) | 18 |
| Build-ignored (AgentSDK) | 3 |
| Build-ignored (Arrow conflicts) | 3 |
| Build-ignored (HANA) | 4 |
| Build-ignored (Cloud SQL) | 2 |
| Build-ignored (Other) | 8 |
| Commented code blocks | 1 file, 22 functions |
| Lint-ignored (future use) | 2 functions |
| Performance profiler disabled | 5 functions |
| **Total** | **38 build-ignored files + 1 file with commented blocks** |

