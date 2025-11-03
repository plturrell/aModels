package search

import (
	"fmt"
	"slices"
	"sync"
	"time"
)

// AgentCatalog summarises the Agent SDK suite/tool inventory.
type AgentCatalog struct {
	Suites []AgentSuite `json:"suites"`
	Tools  []AgentTool  `json:"tools"`
}

// AgentSuite describes a connected MCP suite.
type AgentSuite struct {
	Name           string    `json:"name"`
	ToolNames      []string  `json:"tool_names"`
	ToolCount      int       `json:"tool_count"`
	Implementation string    `json:"implementation"`
	Version        string    `json:"version"`
	AttachedAt     time.Time `json:"attached_at"`
}

// AgentTool describes an individual tool exposed by the Agent SDK runtime.
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

// Normalize ensures derived fields remain consistent.
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

// catalogState manages shared catalog access inside a service.
type catalogState struct {
	mu        sync.RWMutex
	catalog   *AgentCatalog
	updatedAt time.Time
	flightURL string
}

func (s *catalogState) SetFlightAddr(addr string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.flightURL = addr
}

func (s *catalogState) FlightAddr() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.flightURL
}

func (s *catalogState) Update(cat AgentCatalog) {
	cat.Normalize()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.catalog = cat.Clone()
	s.updatedAt = time.Now().UTC()
}

func (s *catalogState) Snapshot() (*AgentCatalog, time.Time) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.catalog == nil {
		return nil, time.Time{}
	}
	return s.catalog.Clone(), s.updatedAt
}

// Enrichment represents catalog enrichment data (stub for AgentSDK dependency)
type Enrichment struct {
	Summary         string
	Prompt          string
	Stats           EnrichmentStats
	Implementations []interface{}
	UniqueTools     []interface{}
	StandaloneTools []interface{}
}

// EnrichmentStats represents statistics about the catalog
type EnrichmentStats struct {
	SuiteCount          int
	UniqueToolCount     int
	ImplementationCount int
	StandaloneToolCount int
	LastAttachmentSuite string
	LastAttachmentAgo   string
}

// EnrichCatalog converts an AgentCatalog into enrichment format
// AgentSDK dependency removed - returns basic enrichment
func EnrichCatalog(cat *AgentCatalog) Enrichment {
	if cat == nil {
		return Enrichment{}
	}
	
	stats := EnrichmentStats{
		SuiteCount:      len(cat.Suites),
		UniqueToolCount: len(cat.Tools),
	}
	
	return Enrichment{
		Summary: fmt.Sprintf("%d suites, %d tools", len(cat.Suites), len(cat.Tools)),
		Stats:   stats,
	}
}
