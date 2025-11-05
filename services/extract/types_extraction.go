package main

// Core types needed across extract service (available regardless of build tags)

// TableProcessSequence represents the sequence/order of table processing.
type TableProcessSequence struct {
	SequenceID   string   `json:"sequence_id"`
	Tables       []string `json:"tables"`
	SourceType   string   `json:"source_type"`
	SourceFile   string   `json:"source_file"`
	SequenceType string   `json:"sequence_type"`
	Order        int      `json:"order"`
}

// TableClassification represents classification info for a table.
type TableClassification struct {
	TableName      string         `json:"table_name"`
	Classification string         `json:"classification"`
	Confidence     float64        `json:"confidence"`
	Evidence       []string       `json:"evidence"`
	Source         string         `json:"source,omitempty"`
	Patterns       []string       `json:"patterns"`
	Props          map[string]any `json:"props,omitempty"`
}
