package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
)

// DomainDetector detects and associates domains with extracted data
type DomainDetector struct {
	domainConfigs map[string]DomainConfig
	localaiURL    string
	mu            sync.RWMutex
	logger        *log.Logger
}

// DomainConfig represents a domain configuration from LocalAI
type DomainConfig struct {
	Name     string   `json:"name"`
	AgentID  string   `json:"agent_id"`
	Keywords []string `json:"keywords"`
	Tags     []string `json:"tags"`
	Layer    string   `json:"layer"`
	Team     string   `json:"team"`
}

// NewDomainDetector creates a new domain detector
func NewDomainDetector(localaiURL string, logger *log.Logger) *DomainDetector {
	dd := &DomainDetector{
		domainConfigs: make(map[string]DomainConfig),
		localaiURL:    localaiURL,
		logger:        logger,
	}
	
	// Load domains on initialization
	if err := dd.LoadDomains(); err != nil {
		logger.Printf("⚠️  Failed to load domains for detection: %v", err)
	}
	
	return dd
}

// LoadDomains loads domain configurations from LocalAI
func (dd *DomainDetector) LoadDomains() error {
	if dd.localaiURL == "" {
		// No LocalAI URL configured, skip domain detection
		return nil
	}
	
	url := strings.TrimSuffix(dd.localaiURL, "/") + "/v1/domains"
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to fetch domains from LocalAI: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("LocalAI returned status %d: %s", resp.StatusCode, string(body))
	}
	
	var domainsResponse struct {
		Data []struct {
			ID     string      `json:"id"`
			Config DomainConfig `json:"config"`
		} `json:"data"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&domainsResponse); err != nil {
		return fmt.Errorf("failed to decode domains response: %w", err)
	}
	
	dd.mu.Lock()
	defer dd.mu.Unlock()
	
	dd.domainConfigs = make(map[string]DomainConfig)
	for _, domain := range domainsResponse.Data {
		config := DomainConfig{
			Name: domain.Name,
		}
		
		// Try direct fields first
		if domain.AgentID != "" {
			config.AgentID = domain.AgentID
		}
		if len(domain.Keywords) > 0 {
			config.Keywords = domain.Keywords
		}
		if len(domain.Tags) > 0 {
			config.Tags = domain.Tags
		}
		if domain.Layer != "" {
			config.Layer = domain.Layer
		}
		if domain.Team != "" {
			config.Team = domain.Team
		}
		
		// Fallback to nested config if direct fields not available
		if domain.Config != nil {
			if config.AgentID == "" {
				if agentID, ok := domain.Config["agent_id"].(string); ok {
					config.AgentID = agentID
				}
			}
			if len(config.Keywords) == 0 {
				if keywords, ok := domain.Config["keywords"].([]interface{}); ok {
					for _, kw := range keywords {
						if kwStr, ok := kw.(string); ok {
							config.Keywords = append(config.Keywords, kwStr)
						}
					}
				}
			}
			if len(config.Tags) == 0 {
				if tags, ok := domain.Config["tags"].([]interface{}); ok {
					for _, tag := range tags {
						if tagStr, ok := tag.(string); ok {
							config.Tags = append(config.Tags, tagStr)
						}
					}
				}
			}
			if config.Layer == "" {
				if layer, ok := domain.Config["layer"].(string); ok {
					config.Layer = layer
				}
			}
			if config.Team == "" {
				if team, ok := domain.Config["team"].(string); ok {
					config.Team = team
				}
			}
		}
		
		dd.domainConfigs[domain.ID] = config
	}
	
	dd.logger.Printf("✅ Loaded %d domains for detection", len(dd.domainConfigs))
	return nil
}

// DetectDomain detects the most appropriate domain for given text content
func (dd *DomainDetector) DetectDomain(text string) (string, string) {
	if dd == nil || len(dd.domainConfigs) == 0 {
		return "", ""
	}
	
	dd.mu.RLock()
	defer dd.mu.RUnlock()
	
	textLower := strings.ToLower(text)
	bestScore := 0
	bestDomain := ""
	bestAgentID := ""
	
	for domainID, config := range dd.domainConfigs {
		if config.AgentID == "" {
			continue
		}
		
		score := 0
		// Check keyword matches
		for _, keyword := range config.Keywords {
			if strings.Contains(textLower, strings.ToLower(keyword)) {
				score++
			}
		}
		
		// Check tag matches (less weight)
		for _, tag := range config.Tags {
			if strings.Contains(textLower, strings.ToLower(tag)) {
				score += 1
			}
		}
		
		if score > bestScore {
			bestScore = score
			bestDomain = domainID
			bestAgentID = config.AgentID
		}
	}
	
	return bestDomain, bestAgentID
}

// AssociateDomainsWithNodes associates domains with extracted nodes based on content
func (dd *DomainDetector) AssociateDomainsWithNodes(nodes []Node) {
	if dd == nil || len(dd.domainConfigs) == 0 {
		return
	}
	
	for i := range nodes {
		node := &nodes[i]
		
		// Build text content from node properties
		text := node.Label
		if node.Type == "table" || node.Type == "column" {
			text += " " + node.ID
		}
		if node.Props != nil {
			if dtype, ok := node.Props["type"].(string); ok {
				text += " " + dtype
			}
			if schema, ok := node.Props["schema"].(string); ok {
				text += " " + schema
			}
		}
		
		// Detect domain
		domainID, agentID := dd.DetectDomain(text)
		if domainID != "" && agentID != "" {
			if node.Props == nil {
				node.Props = make(map[string]any)
			}
			node.Props["agent_id"] = agentID
			node.Props["domain"] = domainID
		}
	}
}

// AssociateDomainsWithEdges associates domains with extracted edges based on source/target nodes
func (dd *DomainDetector) AssociateDomainsWithEdges(edges []Edge, nodes map[string]*Node) {
	if dd == nil || len(dd.domainConfigs) == 0 {
		return
	}
	
	for i := range edges {
		edge := &edges[i]
		
		// Inherit domain from source or target node
		sourceNode, sourceOk := nodes[edge.SourceID]
		targetNode, targetOk := nodes[edge.TargetID]
		
		var agentID, domainID string
		if sourceOk && sourceNode.Props != nil {
			if aid, ok := sourceNode.Props["agent_id"].(string); ok {
				agentID = aid
			}
			if did, ok := sourceNode.Props["domain"].(string); ok {
				domainID = did
			}
		}
		
		// Prefer target node if source doesn't have domain
		if agentID == "" && targetOk && targetNode.Props != nil {
			if aid, ok := targetNode.Props["agent_id"].(string); ok {
				agentID = aid
			}
			if did, ok := targetNode.Props["domain"].(string); ok {
				domainID = did
			}
		}
		
		if agentID != "" && domainID != "" {
			if edge.Props == nil {
				edge.Props = make(map[string]any)
			}
			edge.Props["agent_id"] = agentID
			edge.Props["domain"] = domainID
		}
	}
}

// AssociateDomainsWithSQL associates domains with SQL queries
func (dd *DomainDetector) AssociateDomainsWithSQL(sqlQueries []string) map[string]string {
	result := make(map[string]string)
	
	if dd == nil || len(dd.domainConfigs) == 0 {
		return result
	}
	
	for _, sql := range sqlQueries {
		domainID, agentID := dd.DetectDomain(sql)
		if domainID != "" && agentID != "" {
			result[sql] = agentID
		}
	}
	
	return result
}

