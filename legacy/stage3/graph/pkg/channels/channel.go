package channels

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrStreamClosed is returned when attempting to send on a closed stream.
var ErrStreamClosed = errors.New("channels: stream closed")

// Stream is a concurrency-safe typed channel that mirrors the async stream
// semantics provided by LangGraph's Python implementation. It supports
// context-aware send/receive, graceful shutdown with error propagation, and
// optional non-blocking accessors.
type Stream[T any] struct {
	ch        chan T
	closed    chan struct{}
	closeOnce sync.Once

	mu  sync.RWMutex
	err error
}

// NewStream constructs a buffered stream with the provided capacity. A
// non-positive capacity results in an unbuffered stream.
func NewStream[T any](capacity int) *Stream[T] {
	if capacity < 0 {
		capacity = 0
	}
	return &Stream[T]{
		ch:     make(chan T, capacity),
		closed: make(chan struct{}),
	}
}

// Send publishes a value into the stream. The call blocks until a receiver is
// ready, the stream is closed, or the context is cancelled.
func (s *Stream[T]) Send(ctx context.Context, value T) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = ErrStreamClosed
		}
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.closed:
		return s.Err()
	case s.ch <- value:
		return nil
	}
}

// TrySend attempts to publish a value without blocking. It returns false when
// the stream buffer is full or the stream has been closed.
func (s *Stream[T]) TrySend(value T) (ok bool) {
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	select {
	case s.ch <- value:
		return true
	case <-s.closed:
		return false
	default:
		return false
	}
}

// Recv pulls the next available value from the stream. The bool return
// indicates whether a value was received; when false, the stream has been
// closed and the accompanying error describes the shutdown cause.
func (s *Stream[T]) Recv(ctx context.Context) (T, bool, error) {
	var zero T
	select {
	case <-ctx.Done():
		return zero, false, ctx.Err()
	case val, ok := <-s.ch:
		if !ok {
			return zero, false, s.Err()
		}
		return val, true, nil
	}
}

// TryRecv fetches a value without blocking. It returns false when no value is
// available. If the stream is closed, ok is false and err reports the close
// reason.
func (s *Stream[T]) TryRecv() (value T, ok bool, err error) {
	select {
	case val, open := <-s.ch:
		if !open {
			return value, false, s.Err()
		}
		return val, true, nil
	default:
		return value, false, nil
	}
}

// Close closes the stream and optionally records an error. The first call wins;
// subsequent calls are no-ops. Pending receivers drain remaining buffered
// values before observing the close error.
func (s *Stream[T]) Close(err error) {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		if err == nil {
			err = ErrStreamClosed
		}
		s.err = err
		close(s.closed)
		close(s.ch)
		s.mu.Unlock()
	})
}

// Err returns the stream close error. It is nil until Close is invoked.
func (s *Stream[T]) Err() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.err
}

// Closed reports whether the stream has been closed.
func (s *Stream[T]) Closed() bool {
	select {
	case <-s.closed:
		return true
	default:
		return false
	}
}

// AwaitClose blocks until the stream is closed or the context is cancelled.
// The returned error mirrors Close's supplied error.
func (s *Stream[T]) AwaitClose(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.closed:
		return s.Err()
	}
}

// Chan exposes a read-only view of the underlying channel. Callers must still
// consult Err() after the channel is closed to retrieve the shutdown reason.
func (s *Stream[T]) Chan() <-chan T {
	return s.ch
}

// Deadline returns the time when the stream was closed, or the zero value if it
// is still open. Consumers can use this to enforce deadlines in finished
// streams.
func (s *Stream[T]) Deadline() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.err == nil {
		return time.Time{}
	}
	return time.Now().UTC()
}
