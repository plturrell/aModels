package channels

import (
	"context"
	"errors"
	"sync"
)

// Broadcaster fans values out to multiple subscribed streams. Each subscriber
// receives every published value until it closes or the broadcaster shuts down.
type Broadcaster[T any] struct {
	mu     sync.RWMutex
	subs   map[*Stream[T]]struct{}
	err    error
	closed bool
}

// NewBroadcaster constructs an empty broadcaster.
func NewBroadcaster[T any]() *Broadcaster[T] {
	return &Broadcaster[T]{subs: make(map[*Stream[T]]struct{})}
}

// Subscribe registers a new downstream stream. When the broadcaster is already
// closed, the returned stream is also closed with the broadcaster's error.
func (b *Broadcaster[T]) Subscribe(capacity int) *Stream[T] {
	stream := NewStream[T](capacity)

	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		stream.Close(b.err)
		return stream
	}
	b.subs[stream] = struct{}{}
	return stream
}

// Publish delivers the value to every active subscriber. If the broadcaster is
// closed, Publish returns the close error. Calls block until every subscriber
// accepts the value or the context is cancelled.
func (b *Broadcaster[T]) Publish(ctx context.Context, value T) error {
	b.mu.RLock()
	if b.closed {
		err := b.err
		b.mu.RUnlock()
		if err == nil {
			return ErrStreamClosed
		}
		return err
	}

	streams := make([]*Stream[T], 0, len(b.subs))
	for s := range b.subs {
		streams = append(streams, s)
	}
	b.mu.RUnlock()

	var toRemove []*Stream[T]

	for _, s := range streams {
		if err := s.Send(ctx, value); err != nil {
			if errors.Is(err, ErrStreamClosed) {
				toRemove = append(toRemove, s)
				continue
			}
			return err
		}
	}

	if len(toRemove) > 0 {
		b.mu.Lock()
		for _, s := range toRemove {
			delete(b.subs, s)
		}
		b.mu.Unlock()
	}

	return nil
}

// Close closes the broadcaster and all subscribed streams.
func (b *Broadcaster[T]) Close(err error) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	b.err = err
	streams := make([]*Stream[T], 0, len(b.subs))
	for s := range b.subs {
		streams = append(streams, s)
	}
	b.subs = nil
	b.mu.Unlock()

	for _, s := range streams {
		s.Close(err)
	}
}

// Err reports the close reason when the broadcaster has shut down.
func (b *Broadcaster[T]) Err() error {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.err
}

// Closed indicates whether the broadcaster has been closed.
func (b *Broadcaster[T]) Closed() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.closed
}
