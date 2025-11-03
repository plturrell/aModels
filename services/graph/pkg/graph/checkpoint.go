package graph

import (
	"context"

	"github.com/langchain-ai/langgraph-go/pkg/checkpoint"
)

// Checkpointer abstracts persistence of intermediate state.
type Checkpointer interface {
	Save(ctx context.Context, key string, payload []byte) error
	Load(ctx context.Context, key string) ([]byte, error)
}

// NoopCheckpointer is used when persistence is optional.
type NoopCheckpointer struct{}

func (NoopCheckpointer) Save(context.Context, string, []byte) error { return nil }
func (NoopCheckpointer) Load(context.Context, string) ([]byte, error) {
	return nil, checkpoint.ErrNotFound
}

var _ Checkpointer = (*NoopCheckpointer)(nil)
