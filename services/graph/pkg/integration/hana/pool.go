//go:build hana

package hana

import (
	stubs "github.com/langchain-ai/langgraph-go/pkg/stubs"
)

// NewPoolFromEnv delegates to the shared hanapool implementation to construct a
// connection pool using environment configuration.
func NewPoolFromEnv() (*stubs.Pool, error) {
	return stubs.NewPoolFromEnv()
}
