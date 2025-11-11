// Package stubs provides stubs for missing agenticAiETH dependencies
// This replaces the missing agenticAiETH_layer1_Blockchain/infrastructure/blockchain package
package stubs

import (
	"context"
	"io"
)

// BlockchainOptions configures blockchain resources
type BlockchainOptions struct{}

// BlockchainResources represents blockchain resources
type BlockchainResources struct{}

// Close closes blockchain resources
func (r *BlockchainResources) Close() error {
	return nil
}

// InitBlockchainResources initializes blockchain resources (stub)
func InitBlockchainResources(ctx context.Context, opts BlockchainOptions) (*BlockchainResources, error) {
	return &BlockchainResources{}, nil
}

// Client is a stub for blockchain client
type Client struct{}

// NewClient creates a new stub client
func NewClient() *Client {
	return &Client{}
}

// ReadCloser stub for blockchain streams
type ReadCloser struct {
	io.Reader
	io.Closer
}

