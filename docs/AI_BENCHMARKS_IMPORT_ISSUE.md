# ai_benchmarks Import Path Issue

## Problem

Files disabled with `//go:build ignore` are trying to import packages that don't match the actual codebase structure.

## Root Cause

The root `go.mod` declares `module ai_benchmarks`, so `ai_benchmarks` is the root module. However, the disabled files are importing paths that don't exist:

### Expected Imports (in disabled files):
- `ai_benchmarks/internal/preprocess`
- `ai_benchmarks/internal/registry`
- `ai_benchmarks/internal/localai`
- `ai_benchmarks/internal/lnn`
- `ai_benchmarks/internal/methods`
- `ai_benchmarks/internal/catalog/flightcatalog`
- `ai_benchmarks/benchmarks/arc`
- `ai_benchmarks/benchmarks/boolq`
- etc.

### Actual Code Locations:
- `pkg/preprocess/` (not `internal/preprocess`)
- `pkg/registry/` (not `internal/registry`)
- `pkg/localai/` (not `internal/localai`)
- `pkg/lnn/` (not `internal/lnn`)
- `pkg/methods/` (not `internal/methods`)
- `pkg/catalog/flightcatalog/` (not `internal/catalog/flightcatalog`)
- `testing/benchmarks/arc/` (not `benchmarks/arc`)
- `testing/benchmarks/boolq/` (not `benchmarks/boolq`)
- etc.

## Affected Files

### Command Files (6 files):
1. `tools/cmd/aibench/main.go`
2. `tools/cmd/aibench/tune.go`
3. `tools/cmd/aibench/nash.go`
4. `tools/cmd/factory/main.go`
5. `tools/cmd/calibrate/main.go`
6. `tools/cmd/benchmark-server/main.go`

### Package Files (6 files):
1. `pkg/lnn/tune.go`
2. `pkg/lnn/nash.go`
3. `pkg/localai/enhanced_inference.go`
4. `pkg/models/maths_mcq.go`
5. `pkg/methods/aggregate.go`
6. `pkg/preprocess/preprocess.go`

### Mapper Files (6 files):
1. `tools/scripts/factory/mappers/triviaqa_mapper.go`
2. `tools/scripts/factory/mappers/socialiqa_mapper.go`
3. `tools/scripts/factory/mappers/piqa_mapper.go`
4. `tools/scripts/factory/mappers/hellaswag_mapper.go`
5. `tools/scripts/factory/mappers/boolq_mapper.go`
6. `tools/scripts/factory/mappers/arc_mapper.go`

### Connector Files (1 file):
1. `tools/scripts/factory/connectors/csv_connector.go`

**Total: 19 files** (Note: Some files also have AgentSDK dependencies)

## Solution Options

### Option 1: Update Import Paths (Recommended)
Update all import statements to match actual code locations:

**Changes needed:**
- `ai_benchmarks/internal/preprocess` → `ai_benchmarks/pkg/preprocess`
- `ai_benchmarks/internal/registry` → `ai_benchmarks/pkg/registry`
- `ai_benchmarks/internal/localai` → `ai_benchmarks/pkg/localai`
- `ai_benchmarks/internal/lnn` → `ai_benchmarks/pkg/lnn`
- `ai_benchmarks/internal/methods` → `ai_benchmarks/pkg/methods`
- `ai_benchmarks/internal/catalog/flightcatalog` → `ai_benchmarks/pkg/catalog/flightcatalog`
- `ai_benchmarks/benchmarks/*` → `ai_benchmarks/testing/benchmarks/*`

**Also update files that use these packages:**
- `testing/benchmarks/arc/arc.go` already uses `ai_benchmarks/internal/methods` and `ai_benchmarks/internal/registry`
- Need to update these as well

### Option 2: Create Directory Structure
Create the expected `internal/` and `benchmarks/` directories and move/symlink code:

**Create:**
- `internal/preprocess/` → symlink or move from `pkg/preprocess/`
- `internal/registry/` → symlink or move from `pkg/registry/`
- `internal/localai/` → symlink or move from `pkg/localai/`
- `internal/lnn/` → symlink or move from `pkg/lnn/`
- `internal/methods/` → symlink or move from `pkg/methods/`
- `internal/catalog/flightcatalog/` → symlink or move from `pkg/catalog/flightcatalog/`
- `benchmarks/` → symlink or move from `testing/benchmarks/`

**Note:** This approach may cause issues with Go module resolution and is not recommended.

### Option 3: Remove Files
If these tools are not needed, remove all 19 files.

## Recommendation

**Option 1 (Update Import Paths)** is recommended because:
1. Matches actual code structure
2. No symlinks or file moves needed
3. Cleaner long-term solution
4. Files in `testing/benchmarks/` also need updating

## Implementation Steps

1. **Update import paths in disabled files:**
   - Replace `internal/` with `pkg/`
   - Replace `benchmarks/` with `testing/benchmarks/`

2. **Update import paths in active files:**
   - `testing/benchmarks/arc/arc.go` and other benchmark files
   - Any other files using `ai_benchmarks/internal/*`

3. **Remove `//go:build ignore` tags** from all 19 files

4. **Test compilation:**
   ```bash
   go build ./tools/cmd/aibench/...
   go build ./tools/cmd/factory/...
   go build ./tools/cmd/calibrate/...
   go build ./tools/cmd/benchmark-server/...
   ```

5. **Fix any remaining compilation errors**

## Files That Also Need Import Updates

These files are NOT disabled but use the wrong import paths:
- `testing/benchmarks/arc/arc.go` - uses `ai_benchmarks/internal/methods` and `ai_benchmarks/internal/registry`
- Other benchmark files in `testing/benchmarks/` may have similar issues

## Additional Dependencies

Some files also depend on missing AgentSDK package:
- `tools/cmd/benchmark-server/main.go`
- `pkg/localai/enhanced_inference.go`

These will need the same treatment as the Arrow conflicts (replace with stubs or local implementations).

