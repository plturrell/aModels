package connectors

import "context"

// SourceSchema represents a simplified API schema description used by integrations.
type SourceSchema struct {
	Tables   []SourceTable
	Metadata map[string]any
}

type SourceTable struct {
	Name       string
	PrimaryKey string
	Columns    []SourceColumn
}

type SourceColumn struct {
	Name string
	Type string
}

// Connector defines minimal capabilities needed by graph integrations.
type Connector interface {
	Connect(ctx context.Context, opts map[string]any) error
	Close() error
	ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error)
	DiscoverSchema(ctx context.Context) (*SourceSchema, error)
}
