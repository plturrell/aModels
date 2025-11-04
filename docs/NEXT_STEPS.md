# Next Steps After Quality Metrics & Orphan Detection

## Current Status

✅ **Completed:**
- Quality metrics system implemented
- Orphan detection scripts created
- Property enrichment completed (8,195 columns)
- Graph reconciliation verified (Neo4j ↔ Postgres)
- Training data exported to CSV
- Training config created (configs/rt_sgmi.yaml)
- All changes synced to GitHub

⚠️ **Issues Identified:**
- 8,195 orphan columns (missing HAS_COLUMN edges)
- 37 nodes missing properties (metadata nodes)

## Immediate Next Steps

### 1. Fix Orphan Columns (HIGH PRIORITY)

**Option A: Code-Level Fix (Recommended)**
- Implement `fixOrphanColumns()` in `services/extract/normalization.go`
- See `docs/GLEAN_CATALOG_IMPROVEMENTS_SUMMARY.md` for implementation
- This will prevent future orphan columns during extraction

**Option B: SQL Fix (Quick Fix)**
- Run `scripts/fix_orphan_columns.sh` (may need refinement for complex IDs)
- Or use the SQL queries in `docs/GLEAN_CATALOG_IMPROVEMENTS.md`

**Action**: Choose approach and implement

### 2. Implement Code Improvements

**Files to Modify:**
- `services/extract/normalization.go` - Add orphan fix function
- `services/extract/main.go` - Add validation, type normalization
- `services/extract/ddl.go` - Improve ID generation

**Implementation Guide**: See `services/extract/IMPROVEMENTS_PROPOSED.md`

### 3. Re-run Extraction (After Code Fixes)

```bash
# Re-extract SGMI data to verify fixes
cd /home/aModels
./scripts/run_sgmi_full_graph.sh

# Verify no orphan columns
./scripts/check_orphans.sh

# Check quality metrics
./scripts/run_quality_metrics.sh
```

### 4. Proceed with Training

**Training Setup:**
- ✅ Training data exported: `data/training/extracts/sgmi/`
- ✅ Config created: `configs/rt_sgmi.yaml`
- ⏳ Fix orphan columns first (may affect training data quality)

**Run Training:**
```bash
cd /home/aModels
docker exec -it training-shell bash

# Inside container
python3 tools/scripts/train_relational_transformer.py \
  --config configs/rt_sgmi.yaml \
  --output models/rt_sgmi/
```

## Long-Term Improvements

### 1. Enhance Extraction Process

- **Type Normalization**: Normalize string/STRING, decimal/DECIMAL variations
- **Better Property Extraction**: Extract more metadata from DDL/JSON
- **ID Generation**: Standardize ID format (remove backticks, consistent naming)

### 2. Add Validation Pipeline

- Pre-persistence validation
- Post-extraction quality checks
- Automated orphan detection and fixing

### 3. Monitoring & Reporting

- Regular quality metrics runs
- Automated orphan detection
- Quality trend tracking

## Decision Points

### Should we fix orphan columns before training?

**Option 1: Fix now (Recommended)**
- Pros: Cleaner training data, better model quality
- Cons: Requires code changes and re-extraction
- Time: ~1-2 hours

**Option 2: Proceed with training**
- Pros: Faster start
- Cons: May affect model quality, harder to fix later
- Time: Immediate

**Recommendation**: Fix orphan columns first, then train

### Training Approach

**Option 1: Full Training**
- Use all 23,459 table-column relationships
- Train on complete SGMI dataset
- Time: Several hours (depends on GPU)

**Option 2: Quick Test**
- Use subset of data
- Validate pipeline first
- Time: ~30 minutes

**Recommendation**: Start with quick test, then full training

## Quick Reference Commands

```bash
# Check quality
./scripts/check_all_quality.sh

# Check orphans
./scripts/check_orphans.sh

# Run quality metrics
./scripts/run_quality_metrics.sh

# Reconcile graphs
./scripts/reconcile_graph_to_postgres.sh

# Export training data
./scripts/export_sgmi_for_training.sh

# Explore data
./scripts/explore_sgmi_data.sh
./scripts/explore_data_flows.sh
```

## Questions to Answer

1. **Fix orphan columns now or proceed with training?**
   - Recommendation: Fix now

2. **Implement code-level fixes or use SQL quick fix?**
   - Recommendation: Code-level (prevents future issues)

3. **Full training or quick test first?**
   - Recommendation: Quick test first

4. **When to proceed with training?**
   - After: Orphan columns fixed, quality metrics acceptable

## Success Criteria

Before training:
- ✅ 0 orphan columns
- ✅ < 50 nodes missing properties (metadata nodes OK)
- ✅ Graph sync verified
- ✅ Training data exported
- ✅ Config created

Ready for training when:
- All quality checks pass
- Orphan columns fixed
- Training data validated

