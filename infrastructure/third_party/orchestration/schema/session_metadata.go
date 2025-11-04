package schema

import "context"

// SessionMetadata captures contextual data for logging chain executions.
type SessionMetadata struct {
	SessionID    string
	UserIDHash   string
	PrivacyLevel string
	Operation    string
	LibraryType  string
}

type sessionMetadataKey struct{}

// WithSessionMetadata annotates a context with metadata used by telemetry handlers.
func WithSessionMetadata(ctx context.Context, meta SessionMetadata) context.Context {
	return context.WithValue(ctx, sessionMetadataKey{}, meta)
}

// SessionMetadataFromContext retrieves metadata from a context, returning zero values when absent.
func SessionMetadataFromContext(ctx context.Context) SessionMetadata {
	if ctx == nil {
		return SessionMetadata{}
	}
	if v := ctx.Value(sessionMetadataKey{}); v != nil {
		if meta, ok := v.(SessionMetadata); ok {
			return meta
		}
	}
	return SessionMetadata{}
}
