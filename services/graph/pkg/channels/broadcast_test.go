package channels_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/channels"
)

func TestBroadcasterPublishToSubscribers(t *testing.T) {
	b := channels.NewBroadcaster[int]()
	sub1 := b.Subscribe(4)
	sub2 := b.Subscribe(4)

	ctx := context.Background()
	if err := b.Publish(ctx, 1); err != nil {
		t.Fatalf("Publish failed: %v", err)
	}
	if err := b.Publish(ctx, 2); err != nil {
		t.Fatalf("Publish failed: %v", err)
	}

	assertRecv := func(stream *channels.Stream[int], expected ...int) {
		for _, want := range expected {
			val, ok, err := stream.Recv(ctx)
			if err != nil || !ok || val != want {
				t.Fatalf("expected %d, got val=%d ok=%v err=%v", want, val, ok, err)
			}
		}
	}
	assertRecv(sub1, 1, 2)
	assertRecv(sub2, 1, 2)
}

func TestBroadcasterClose(t *testing.T) {
	b := channels.NewBroadcaster[string]()
	sub := b.Subscribe(1)
	closeErr := errors.New("done")
	b.Close(closeErr)

	if !b.Closed() {
		t.Fatalf("broadcaster should be closed")
	}
	if !errors.Is(b.Err(), closeErr) {
		t.Fatalf("unexpected close error: %v", b.Err())
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, ok, err := sub.Recv(ctx)
	if ok || !errors.Is(err, closeErr) {
		t.Fatalf("expected close error propagated, got ok=%v err=%v", ok, err)
	}

	// new subscription after close should be immediately closed.
	late := b.Subscribe(1)
	if !late.Closed() {
		t.Fatalf("late subscriber should be closed")
	}
	if err := late.AwaitClose(ctx); !errors.Is(err, closeErr) {
		t.Fatalf("expected closeErr, got %v", err)
	}
}

func TestBroadcasterPublishAfterSubscriberClose(t *testing.T) {
	b := channels.NewBroadcaster[int]()
	sub := b.Subscribe(1)
	sub.Close(nil)

	if err := b.Publish(context.Background(), 1); err != nil {
		t.Fatalf("Publish failed when subscriber closed: %v", err)
	}
}

func TestBroadcasterPublishAfterClose(t *testing.T) {
	b := channels.NewBroadcaster[int]()
	b.Close(nil)
	if err := b.Publish(context.Background(), 1); !errors.Is(err, channels.ErrStreamClosed) {
		t.Fatalf("expected ErrStreamClosed, got %v", err)
	}
}
