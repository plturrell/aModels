package main

// Core types needed across extract service (available regardless of build tags)
// These types are shared across multiple files in the extract service.

// TableProcessSequence represents the sequence/order of table processing.
// It tracks the order in which tables are processed, extracted from SQL queries,
// Control-M files, DDL statements, or other sources.
type TableProcessSequence struct {
	SequenceID   string   `json:"sequence_id"`   // Unique identifier for this sequence
	Tables       []string `json:"tables"`        // Ordered list of tables in processing sequence
	SourceType   string   `json:"source_type"`   // sql, controlm, ddl, etc.
	SourceFile   string   `json:"source_file"`   // Source file or identifier
	SequenceType string   `json:"sequence_type"` // insert, update, select, cte, etc.
	Order        int      `json:"order"`         // Processing order (0-based)
}

// TableClassification classifies a table as transaction, reference, staging, etc.
// Phase 5: Extended with Props for quality scores and review flags.
// The Source field indicates where the classification came from (e.g., "ddl_0", "json_1").
type TableClassification struct {
	TableName      string         `json:"table_name"`      // Name of the table
	Classification string         `json:"classification"`  // transaction, reference, lookup, staging, test, unknown
	Confidence     float64        `json:"confidence"`      // 0.0 to 1.0 confidence score
	Evidence       []string       `json:"evidence"`        // Reasons for classification
	Source         string         `json:"source,omitempty"` // Source identifier (e.g., "ddl_0", "json_1")
	Patterns       []string       `json:"patterns"`        // Patterns that led to classification
	Props          map[string]any `json:"props,omitempty"` // Additional properties (quality_score, needs_review, etc.)
}
