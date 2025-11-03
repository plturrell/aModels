package maths

import (
	agentmaths "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

// Provider exposes the configurable maths backend used by agenticAiETH. LangGraph-Go
// builds atop the same primitives to remain interoperable with existing agents.
type Provider = agentmaths.Provider

// NewProvider constructs the default maths provider leveraging the upstream
// implementation.
func NewProvider() Provider {
	return agentmaths.New()
}
