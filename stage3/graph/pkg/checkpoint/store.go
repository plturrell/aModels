package checkpoint

import (
	"context"
	"errors"
)

// ErrNotFound is returned when no checkpoint exists for the supplied key.
var ErrNotFound = errors.New("checkpoint: not found")

// Store is the storage contract for saving and restoring graph execution state.
type Store interface {
	Save(ctx context.Context, key string, payload []byte) error
	Load(ctx context.Context, key string) ([]byte, error)
	Delete(ctx context.Context, key string) error
}
