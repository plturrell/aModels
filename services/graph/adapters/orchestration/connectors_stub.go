package orchestration

import (
    "context"
    "log"

    ports "github.com/langchain-ai/langgraph-go/pkg/connectors"
)

type stubConn struct{}

func (s *stubConn) Connect(ctx context.Context, opts map[string]any) error { return nil }
func (s *stubConn) Close() error { return nil }
func (s *stubConn) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) {
    return []map[string]any{}, nil
}
func (s *stubConn) DiscoverSchema(ctx context.Context) (*ports.SourceSchema, error) {
    return &ports.SourceSchema{Tables: []any{}, Metadata: map[string]any{}}, nil
}

type Factory struct{}

func NewFactory() *Factory { return &Factory{} }

func (f *Factory) NewMurexConnector(config map[string]any, logger *log.Logger) ports.DataConnector { return &stubConn{} }
func (f *Factory) NewAxiomConnector(config map[string]any, logger *log.Logger) ports.DataConnector { return &stubConn{} }
func (f *Factory) NewSAPGLConnector(config map[string]any, logger *log.Logger) ports.DataConnector { return &stubConn{} }
func (f *Factory) NewRCOConnector(config map[string]any, logger *log.Logger) ports.DataConnector { return &stubConn{} }
func (f *Factory) NewBCRSConnector(config map[string]any, logger *log.Logger) ports.DataConnector { return &stubConn{} }


