package search

import (
	"fmt"
	"strings"
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

// cloneStringSlice creates a copy of a string slice (Go 1.18 compatible replacement for slices.Clone)
func cloneStringSlice(src []string) []string {
	if src == nil {
		return nil
	}
	dst := make([]string, len(src))
	copy(dst, src)
	return dst
}

// cloneAgentToolSlice creates a copy of an AgentTool slice (Go 1.18 compatible replacement for slices.Clone)
func cloneAgentToolSlice(src []AgentTool) []AgentTool {
	if src == nil {
		return nil
	}
	dst := make([]AgentTool, len(src))
	copy(dst, src)
	return dst
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
		out.Suites[i].ToolNames = cloneStringSlice(suite.ToolNames)
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
		c.Suites[i].ToolNames = cloneStringSlice(c.Suites[i].ToolNames)
		c.Suites[i].ToolCount = len(c.Suites[i].ToolNames)
	}
	c.Tools = cloneAgentToolSlice(c.Tools)
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
	SuiteCount          int    `json:"suite_count"`
	UniqueToolCount     int    `json:"unique_tool_count"`
	ImplementationCount int    `json:"implementation_count"`
	StandaloneToolCount int    `json:"standalone_tool_count"`
	LastAttachmentSuite string `json:"last_attachment_suite"`
	LastAttachmentAgo   string `json:"last_attachment_ago"`
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

	implementationSet := map[string]struct{}{}
	var latestSuite string
	var latestTime time.Time
	var suiteLines []string

	for _, suite := range cat.Suites {
		if suite.Implementation != "" {
			implementationSet[suite.Implementation] = struct{}{}
		}
		if suite.AttachedAt.After(latestTime) {
			latestTime = suite.AttachedAt
			latestSuite = suite.Name
		}
		suiteLines = append(suiteLines, fmt.Sprintf("- %s (%d tools)", suite.Name, len(suite.ToolNames)))
	}

	stats.ImplementationCount = len(implementationSet)

	if stats.UniqueToolCount > 0 {
		stats.StandaloneToolCount = stats.UniqueToolCount
	}
	if !latestTime.IsZero() {
		stats.LastAttachmentSuite = latestSuite
		stats.LastAttachmentAgo = time.Since(latestTime).Round(time.Minute).String()
	}

	uniqueTools := make([]interface{}, 0, len(cat.Tools))
	standalone := make([]interface{}, 0, len(cat.Tools))
	for _, tool := range cat.Tools {
		uniqueTools = append(uniqueTools, tool.Name)
		standalone = append(standalone, map[string]string{
			"name":        tool.Name,
			"description": tool.Description,
		})
	}

	promptBuilder := &strings.Builder{}
	promptBuilder.WriteString("Suites:\n")
	for _, line := range suiteLines {
		promptBuilder.WriteString(line)
		promptBuilder.WriteByte('\n')
	}
	if len(cat.Tools) > 0 {
		promptBuilder.WriteString("Tools:\n")
		for _, tool := range cat.Tools {
			promptBuilder.WriteString(fmt.Sprintf("- %s\n", tool.Name))
		}
	}

	summary := fmt.Sprintf("%d suites, %d tools", len(cat.Suites), len(cat.Tools))

	return Enrichment{
		Summary:         summary,
		Prompt:          promptBuilder.String(),
		Stats:           stats,
		Implementations: make([]interface{}, 0, stats.ImplementationCount),
		UniqueTools:     uniqueTools,
		StandaloneTools: standalone,
	}
}
