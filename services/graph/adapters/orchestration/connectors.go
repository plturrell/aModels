//go:build adapters_orch
// +build adapters_orch
package orchestration

import (
    "context"
    "log"

    orch "github.com/plturrell/aModels/services/orchestration/agents/connectors"
    ports "github.com/langchain-ai/langgraph-go/pkg/connectors"
)

type dataConnector struct{
    impl interface{
        Connect(ctx context.Context, opts map[string]any) error
        Close() error
        ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error)
        DiscoverSchema(ctx context.Context) (*struct{Tables []any; Metadata map[string]any}, error)
    }
}

func (d *dataConnector) Connect(ctx context.Context, opts map[string]any) error { return d.impl.Connect(ctx, opts) }
func (d *dataConnector) Close() error { return d.impl.Close() }
func (d *dataConnector) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) { return d.impl.ExtractData(ctx, query) }
func (d *dataConnector) DiscoverSchema(ctx context.Context) (*ports.SourceSchema, error) {
    s, err := d.impl.DiscoverSchema(ctx)
    if err != nil { return nil, err }
    if s == nil { return nil, nil }
    return &ports.SourceSchema{Tables: s.Tables, Metadata: s.Metadata}, nil
}

type Factory struct{}

func NewFactory() *Factory { return &Factory{} }

func (f *Factory) NewMurexConnector(config map[string]any, logger *log.Logger) ports.DataConnector {
    return &dataConnector{impl: orch.NewMurexConnector(config, logger)}
}
func (f *Factory) NewAxiomConnector(config map[string]any, logger *log.Logger) ports.DataConnector {
    return &dataConnector{impl: orch.NewAxiomConnector(config, logger)}
}
func (f *Factory) NewSAPGLConnector(config map[string]any, logger *log.Logger) ports.DataConnector {
    return &dataConnector{impl: orch.NewSAPGLConnector(config, logger)}
}
func (f *Factory) NewRCOConnector(config map[string]any, logger *log.Logger) ports.DataConnector {
    return &dataConnector{impl: orch.NewRCOConnector(config, logger)}
}
func (f *Factory) NewBCRSConnector(config map[string]any, logger *log.Logger) ports.DataConnector {
    return &dataConnector{impl: orch.NewBCRSConnector(config, logger)}
}


