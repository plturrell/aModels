package channels

import (
	"context"
	"time"
)

// Message represents a typed payload that flows between nodes.
type Message[T any] struct {
	Value   T
	Meta    map[string]any
	Created time.Time
}

// Request is a bidirectional message that carries a response stream.
type Request[T any, R any] struct {
	Payload  T
	Response *Stream[Message[R]]
	Meta     map[string]any
	Created  time.Time
}

// NewMessage creates a message with the current timestamp.
func NewMessage[T any](value T) Message[T] {
	return Message[T]{
		Value:   value,
		Meta:    make(map[string]any),
		Created: time.Now().UTC(),
	}
}

// NewRequest constructs a request with an associated response stream.
func NewRequest[T any, R any](payload T, respBuffer int) Request[T, R] {
	return Request[T, R]{
		Payload:  payload,
		Response: NewStream[Message[R]](respBuffer),
		Meta:     make(map[string]any),
		Created:  time.Now().UTC(),
	}
}

// SendResponse publishes a response message onto the request's response stream.
func (r Request[T, R]) SendResponse(ctx context.Context, msg Message[R]) error {
	if r.Response == nil {
		return ErrStreamClosed
	}
	return r.Response.Send(ctx, msg)
}

// CloseResponses closes the response stream with an optional error.
func (r Request[T, R]) CloseResponses(err error) {
	if r.Response != nil {
		r.Response.Close(err)
	}
}
