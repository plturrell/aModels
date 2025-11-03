package postgresgrpc

import (
	"context"
	"fmt"

	postgresv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/gen/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client wraps the Postgres LangService gRPC API.
type Client struct {
	conn *grpc.ClientConn
	svc  postgresv1.PostgresLangServiceClient
}

// Dial initialises the gRPC client with optional transport options.
func Dial(ctx context.Context, addr string, opts ...grpc.DialOption) (*Client, error) {
	if addr == "" {
		return nil, fmt.Errorf("postgres gRPC address is required")
	}
	if len(opts) == 0 {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("dial postgres gRPC: %w", err)
	}
	return &Client{
		conn: conn,
		svc:  postgresv1.NewPostgresLangServiceClient(conn),
	}, nil
}

// Close releases the underlying connection.
func (c *Client) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// HealthCheck proxies the health RPC.
func (c *Client) HealthCheck(ctx context.Context) (*postgresv1.HealthCheckResponse, error) {
	if c == nil {
		return nil, fmt.Errorf("postgres client not initialised")
	}
	resp, err := c.svc.HealthCheck(ctx, &postgresv1.HealthCheckRequest{})
	if err != nil {
		return nil, fmt.Errorf("health check: %w", err)
	}
	return resp, nil
}

// ListOperations fetches operations using the provided request (nil uses defaults).
func (c *Client) ListOperations(ctx context.Context, req *postgresv1.ListLangOperationsRequest) (*postgresv1.ListLangOperationsResponse, error) {
	if c == nil {
		return nil, fmt.Errorf("postgres client not initialised")
	}
	if req == nil {
		req = &postgresv1.ListLangOperationsRequest{}
	}
	resp, err := c.svc.ListLangOperations(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("list operations: %w", err)
	}
	return resp, nil
}

// GetAnalytics invokes the analytics RPC.
func (c *Client) GetAnalytics(ctx context.Context, req *postgresv1.AnalyticsRequest) (*postgresv1.AnalyticsResponse, error) {
	if c == nil {
		return nil, fmt.Errorf("postgres client not initialised")
	}
	if req == nil {
		req = &postgresv1.AnalyticsRequest{}
	}
	resp, err := c.svc.GetAnalytics(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("get analytics: %w", err)
	}
	return resp, nil
}
