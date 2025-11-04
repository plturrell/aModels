# Orphan Columns Analysis

## Issue Summary

**8,195 columns** are missing `HAS_COLUMN` edges to their parent tables.

## Root Cause Analysis

### Pattern Analysis

Looking at orphan column IDs:
- `sgmi_scb_product_dat`.`level0` - Has backticks in ID
- `vw_sgmi_scb_product_dat.sgmi_scb_product_dat`.`prod_desc` - Nested view structure
- Column IDs contain backticks: `` `column_name` ``
- Table IDs may or may not have backticks

### Why This Happens

1. **View Columns**: View columns reference base table columns but may not have direct `HAS_COLUMN` edges
2. **ID Format Mismatch**: Column IDs and table IDs use different formatting (with/without backticks)
3. **Extraction Process**: The extraction process may create columns from views without creating proper table relationships
4. **Normalization**: Normalization may drop edges if table IDs don't match exactly

### Affected Tables

Top tables with orphan columns:
- `sgmi_all_f`: 827 orphan columns
- `vw_sgmi_all_f`: 821 orphan columns  
- `v_exposure_retail`: 479 orphan columns
- `sgmi_exposure_f`: 479 orphan columns
- `sgmi_csr_f`: 408 orphan columns

## Solutions

### Immediate Fix (SQL)

A comprehensive SQL fix was attempted but needs refinement due to complex ID patterns.

### Code-Level Fixes

See `services/extract/IMPROVEMENTS_PROPOSED.md` for:
1. Adding orphan column fix to normalization
2. Improving ID matching logic
3. Adding validation warnings

### Long-Term Solution

1. **Improve ID Generation**: Standardize ID format across all extraction sources
2. **Better View Handling**: Properly link view columns to base table columns
3. **Validation**: Add pre-persistence validation to catch orphan columns
4. **Auto-Fix**: Add automatic orphan column fixing in normalization

## Impact

- **Data Quality**: Orphan columns reduce data quality
- **Training**: May affect training data completeness
- **Queries**: Makes it harder to query table-column relationships

## Recommendations

1. ✅ **Fix orphan columns** before training (use SQL fix or code fix)
2. ✅ **Add validation** to prevent future orphan columns
3. ✅ **Improve extraction** to create proper relationships from the start
4. ✅ **Monitor** for orphan columns in future extractions

