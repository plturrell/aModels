package channels_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/channels"
)

func TestStreamSendRecv(t *testing.T) {
	stream := channels.NewStream[int](1)
	ctx := context.Background()

	if err := stream.Send(ctx, 42); err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	val, ok, err := stream.Recv(ctx)
	if err != nil {
		t.Fatalf("Recv error: %v", err)
	}
	if !ok {
		t.Fatalf("expected ok=true")
	}
	if val != 42 {
		t.Fatalf("unexpected value: %d", val)
	}
}

func TestStreamCloseWithError(t *testing.T) {
	stream := channels.NewStream[string](0)
	stream.Close(errors.New("boom"))

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	_, ok, err := stream.Recv(ctx)
	if ok {
		t.Fatalf("expected ok=false after close")
	}
	if err == nil || err.Error() != "boom" {
		t.Fatalf("expected close error, got %v", err)
	}

	if stream.Err() == nil {
		t.Fatalf("Err should return close error")
	}
}

func TestStreamSendAfterClose(t *testing.T) {
	stream := channels.NewStream[int](1)
	stream.Close(nil)

	err := stream.Send(context.Background(), 1)
	if !errors.Is(err, channels.ErrStreamClosed) {
		t.Fatalf("expected ErrStreamClosed, got %v", err)
	}
}

func TestStreamTrySendAndTryRecv(t *testing.T) {
	stream := channels.NewStream[int](1)
	if ok := stream.TrySend(7); !ok {
		t.Fatalf("TrySend should succeed with room in buffer")
	}

	if ok := stream.TrySend(8); ok {
		t.Fatalf("TrySend should fail when buffer full")
	}

	val, ok, err := stream.TryRecv()
	if err != nil || !ok || val != 7 {
		t.Fatalf("TryRecv unexpected result: val=%d ok=%v err=%v", val, ok, err)
	}

	// nothing left, should return ok=false, err=nil
	_, ok, err = stream.TryRecv()
	if ok || err != nil {
		t.Fatalf("expected empty TryRecv to return ok=false, err=nil")
	}
}

func TestStreamContextCancellation(t *testing.T) {
	stream := channels.NewStream[int](0)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	err := stream.Send(ctx, 1)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context cancellation, got %v", err)
	}
}

func TestStreamAwaitClose(t *testing.T) {
	stream := channels.NewStream[int](0)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(20 * time.Millisecond)
		stream.Close(nil)
	}()

	if err := stream.AwaitClose(ctx); !errors.Is(err, channels.ErrStreamClosed) {
		t.Fatalf("expected ErrStreamClosed, got %v", err)
	}
	wg.Wait()
}
