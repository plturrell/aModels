package extractgrpc

import (
	"context"
	"fmt"

	extractpb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract/gen/extractpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client wraps a gRPC connection to the extract service.
type Client struct {
	conn *grpc.ClientConn
	svc  extractpb.ExtractServiceClient
}

// Dial initialises a gRPC client to the extract service using insecure transport by default.
func Dial(ctx context.Context, addr string, opts ...grpc.DialOption) (*Client, error) {
	if addr == "" {
		return nil, fmt.Errorf("extract gRPC address is required")
	}
	if len(opts) == 0 {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("dial extract gRPC: %w", err)
	}
	return &Client{
		conn: conn,
		svc:  extractpb.NewExtractServiceClient(conn),
	}, nil
}

// Close terminates the underlying gRPC connection.
func (c *Client) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// Extract invokes the extract service with the supplied document payload.
func (c *Client) Extract(ctx context.Context, document string) (*extractpb.ExtractResponse, error) {
	if c == nil {
		return nil, fmt.Errorf("extract client not initialised")
	}
	req := &extractpb.ExtractRequest{
		Document: document,
	}
	resp, err := c.svc.Extract(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("extract rpc: %w", err)
	}
	return resp, nil
}
