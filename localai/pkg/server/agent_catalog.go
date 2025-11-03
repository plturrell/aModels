package server

import (
	"slices"
	"time"
)

// AgentCatalog captures the MCP suite/tool metadata from the Agent SDK Flight server.
type AgentCatalog struct {
	Suites []AgentSuite `json:"suites"`
	Tools  []AgentTool  `json:"tools"`
}

// AgentSuite summarises a connected MCP suite.
type AgentSuite struct {
	Name           string    `json:"name"`
	ToolNames      []string  `json:"tool_names"`
	ToolCount      int       `json:"tool_count"`
	Implementation string    `json:"implementation"`
	Version        string    `json:"version"`
	AttachedAt     time.Time `json:"attached_at"`
}

// AgentTool summarises an individual tool exposed to the agent runtime.
type AgentTool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// Clone returns a deep copy of the catalog.
func (c *AgentCatalog) Clone() *AgentCatalog {
	if c == nil {
		return nil
	}
	out := &AgentCatalog{
		Suites: make([]AgentSuite, len(c.Suites)),
		Tools:  make([]AgentTool, len(c.Tools)),
	}
	for i, suite := range c.Suites {
		out.Suites[i] = suite
		out.Suites[i].ToolNames = slices.Clone(suite.ToolNames)
	}
	copy(out.Tools, c.Tools)
	return out
}

// Normalize ensures derived fields are consistent.
func (c *AgentCatalog) Normalize() {
	if c == nil {
		return
	}
	for i := range c.Suites {
		c.Suites[i].ToolNames = slices.Clone(c.Suites[i].ToolNames)
		c.Suites[i].ToolCount = len(c.Suites[i].ToolNames)
	}
	c.Tools = slices.Clone(c.Tools)
}
