
package aiplatform

import (
	"context"

	"github.com/langchain-ai/langgraph-go/pkg/integration/aiplatform"
)

// Store is a checkpoint store that uses the AI Platform service.
type Store struct {
	client *aiplatform.Client
}

// NewStore creates a new AI Platform checkpoint store.
func NewStore(client *aiplatform.Client) *Store {
	return &Store{
		client: client,
	}
}

// Save saves a checkpoint.
func (s *Store) Save(ctx context.Context, checkpointID string, data []byte) error {
	return s.client.SaveCheckpoint(ctx, checkpointID, data)
}

// Load loads a checkpoint.
func (s *Store) Load(ctx context.Context, checkpointID string) ([]byte, error) {
	return s.client.LoadCheckpoint(ctx, checkpointID)
}

// Delete is not implemented.
func (s *Store) Delete(ctx context.Context, checkpointID string) error {
	return nil
}
