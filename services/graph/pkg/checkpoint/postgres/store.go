package postgres

import (
	"context"
	"database/sql"

	_ "github.com/lib/pq"

	"github.com/langchain-ai/langgraph-go/pkg/checkpoint"
)

// Store persists checkpoints inside PostgreSQL. The implementation is a stub
// that will be filled in during the port.
type Store struct {
	db *sql.DB
}

// NewStore constructs the Store using an existing sql.DB handle.
func NewStore(db *sql.DB) *Store {
	return &Store{db: db}
}

// Save implements checkpoint.Store.
func (s *Store) Save(ctx context.Context, key string, payload []byte) error {
	return nil
}

// Load implements checkpoint.Store.
func (s *Store) Load(ctx context.Context, key string) ([]byte, error) {
	return nil, nil
}

// Delete implements checkpoint.Store.
func (s *Store) Delete(ctx context.Context, key string) error {
	return nil
}

var _ checkpoint.Store = (*Store)(nil)
