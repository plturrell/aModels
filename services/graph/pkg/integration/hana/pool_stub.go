//go:build !hana

package hana

import (
	"errors"

	stubs "github.com/langchain-ai/langgraph-go/pkg/stubs"
)

var errHANABuildTagRequired = errors.New("hana integration: build with -tags hana to enable HANA support")

// NewPoolFromEnv returns an error when the binary is built without the hana tag.
func NewPoolFromEnv() (*stubs.Pool, error) {
	return nil, errHANABuildTagRequired
}
