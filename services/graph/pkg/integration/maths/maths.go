package maths

import (
	stubs "github.com/langchain-ai/langgraph-go/pkg/stubs"
)

// Provider exposes the configurable maths backend used by agenticAiETH. LangGraph-Go
// builds atop the same primitives to remain interoperable with existing agents.
type Provider = stubs.Provider

// NewProvider constructs the default maths provider leveraging the upstream
// implementation.
func NewProvider() Provider {
	return stubs.New()
}
