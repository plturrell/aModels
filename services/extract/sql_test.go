package main

import "testing"

func TestParseSQLInsertJoins(t *testing.T) {
	sql := `INSERT INTO target_table SELECT a.col1 AS col1, b.col2 FROM source_table a JOIN other_table b ON a.id = b.id`
	lineage, err := parseSQL(sql)
	if err != nil {
		t.Fatalf("parseSQL: %v", err)
	}

	if len(lineage.TargetTables) != 1 || lineage.TargetTables[0] != "target_table" {
		t.Fatalf("unexpected target tables: %#v", lineage.TargetTables)
	}

	if len(lineage.SourceTables) != 2 {
		t.Fatalf("expected two source tables, got %#v", lineage.SourceTables)
	}

	if len(lineage.ColumnLineage) == 0 {
		t.Fatalf("expected column lineage entries")
	}
}

func TestParseSQLFallbackSchemaQualified(t *testing.T) {
	sql := `INSERT INTO dbo.target SELECT col1, col2 FROM dbo.source`

	matches := insertTargetRegex.FindAllStringSubmatch(sql, -1)
	if len(matches) == 0 || len(matches[0]) < 2 {
		t.Fatalf("insertTargetRegex failed to match schema qualified identifier")
	}
	t.Logf("regex captures: %#v", matches[0][1])

	lineage, err := parseSQL(sql)
	if err != nil {
		t.Fatalf("parseSQL fallback: %v", err)
	}

	fallback := fallbackParseSQL(sql)
	if fallback == nil {
		t.Fatalf("fallbackParseSQL returned nil")
	}
	t.Logf("fallback targets: %#v sources: %#v", fallback.TargetTables, fallback.SourceTables)
	t.Logf("targets: %#v sources: %#v", lineage.TargetTables, lineage.SourceTables)

	if len(lineage.TargetTables) != 1 || lineage.TargetTables[0] != "dbo.target" {
		t.Fatalf("unexpected target tables: %#v", lineage.TargetTables)
	}

	if len(lineage.SourceTables) != 1 || lineage.SourceTables[0] != "dbo.source" {
		t.Fatalf("unexpected source tables: %#v", lineage.SourceTables)
	}

	if len(lineage.ColumnLineage) == 0 {
		t.Fatalf("expected fallback column lineage entries")
	}
}
