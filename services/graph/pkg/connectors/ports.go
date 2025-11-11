package connectors

import (
    "context"
    "log"
)

// SourceSchema is a minimal schema description sufficient for graph integrations.
type SourceSchema struct {
    Tables   []any
    Metadata map[string]any
}

// DataConnector defines the minimal contract graph integrations need.
type DataConnector interface {
    Connect(ctx context.Context, opts map[string]any) error
    Close() error
    ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error)
    DiscoverSchema(ctx context.Context) (*SourceSchema, error)
}

// Factory abstracts connector construction; implemented by adapters.
type Factory interface {
    NewMurexConnector(config map[string]any, logger *log.Logger) DataConnector
    NewAxiomConnector(config map[string]any, logger *log.Logger) DataConnector
    NewSAPGLConnector(config map[string]any, logger *log.Logger) DataConnector
    NewRCOConnector(config map[string]any, logger *log.Logger) DataConnector
    NewBCRSConnector(config map[string]any, logger *log.Logger) DataConnector
}


