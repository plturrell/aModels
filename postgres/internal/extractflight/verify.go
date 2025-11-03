package extractflight

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Verify connects to the extract Flight endpoint and logs the number of nodes and edges fetched.
func Verify(parent context.Context, addr string, logger *slog.Logger) {
	if strings.TrimSpace(addr) == "" {
		return
	}

	ctx, cancel := context.WithTimeout(parent, 10*time.Second)
	defer cancel()

	if logger == nil {
		logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
	}

	client, err := flight.NewClientWithMiddlewareCtx(ctx, addr, nil, nil, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logger.Error("failed to dial extract flight", "error", err)
		return
	}
	defer client.Close()

	nodes, err := fetchCount(ctx, client, []string{"graph", "nodes"})
	if err != nil {
		logger.Error("failed to fetch graph nodes", "error", err)
		return
	}

	edges, err := fetchCount(ctx, client, []string{"graph", "edges"})
	if err != nil {
		logger.Error("failed to fetch graph edges", "error", err)
		return
	}

	logger.Info("extract flight verified", "nodes", nodes, "edges", edges)
}

func fetchCount(ctx context.Context, client flight.Client, path []string) (int64, error) {
	descriptor := &flight.FlightDescriptor{Type: flight.DescriptorPATH, Path: path}

	info, err := client.GetFlightInfo(ctx, descriptor)
	if err != nil {
		return 0, fmt.Errorf("get flight info: %w", err)
	}
	if len(info.Endpoint) == 0 {
		return 0, nil
	}

	alloc := memory.NewGoAllocator()
	var total int64

	for _, endpoint := range info.Endpoint {
		stream, err := client.DoGet(ctx, endpoint.Ticket)
		if err != nil {
			return 0, fmt.Errorf("flight doget: %w", err)
		}

		reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(alloc))
		if err != nil {
			stream.CloseSend()
			return 0, fmt.Errorf("create reader: %w", err)
		}

		for reader.Next() {
			record := reader.Record()
			total += record.NumRows()
		}
		reader.Release()
		_ = stream.CloseSend()
	}

	return total, nil
}
