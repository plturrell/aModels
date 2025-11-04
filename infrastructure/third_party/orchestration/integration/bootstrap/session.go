package bootstrap

import (
	"context"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
)

// SessionMetadata is an alias for schema.SessionMetadata exposed for convenience.
type SessionMetadata = schema.SessionMetadata

// WithSessionMetadata annotates a context with telemetry metadata.
func WithSessionMetadata(ctx context.Context, meta SessionMetadata) context.Context {
	return schema.WithSessionMetadata(ctx, meta)
}

// SessionMetadataFromContext retrieves telemetry metadata from a context.
func SessionMetadataFromContext(ctx context.Context) SessionMetadata {
	return schema.SessionMetadataFromContext(ctx)
}
