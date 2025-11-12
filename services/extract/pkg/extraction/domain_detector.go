package extraction

import (
	"context"
	"log"

	"ai_benchmarks/services/shared/pkg/domain"
)

// DomainDetector wraps the shared domain.Detector so existing call sites remain unchanged.
type DomainDetector struct {
	detector *domain.Detector
	logger   *log.Logger
}

// Config returns the domain configuration for the given domain ID.
func (dd *DomainDetector) Config(domainID string) (domain.DomainConfig, bool) {
	if dd == nil || dd.detector == nil {
		return domain.DomainConfig{}, false
	}
	return dd.detector.Config(domainID)
}

// DomainCount returns the number of loaded domains.
func (dd *DomainDetector) DomainCount() int {
	if dd == nil || dd.detector == nil {
		return 0
	}
	return dd.detector.DomainCount()
}

// Domains returns a snapshot of loaded domain configurations.
func (dd *DomainDetector) Domains() map[string]domain.DomainConfig {
	if dd == nil || dd.detector == nil {
		return map[string]domain.DomainConfig{}
	}
	return dd.detector.Domains()
}

// NewDomainDetector creates a new domain detector backed by the shared package.
func NewDomainDetector(localaiURL string, logger *log.Logger) *DomainDetector {
	det := domain.NewDetector(localaiURL, logger)
	if err := det.LoadDomains(context.Background()); err != nil && err != domain.ErrNoDomains {
		logger.Printf("⚠️  Failed to load domains for detection: %v", err)
	}
	return &DomainDetector{detector: det, logger: logger}
}

// LoadDomains loads domain configurations from LocalAI
func (dd *DomainDetector) LoadDomains() error {
	if dd == nil || dd.detector == nil {
		return nil
	}
	return dd.detector.LoadDomains(context.Background())
}

// DetectDomain detects the most appropriate domain for given text content
func (dd *DomainDetector) DetectDomain(text string) (string, string) {
	return dd.detector.DetectDomain(text)
}

// AssociateDomainsWithNodes associates domains with extracted nodes based on content
func (dd *DomainDetector) AssociateDomainsWithNodes(nodes []Node) {
	if dd == nil || dd.detector == nil {
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
	if dd == nil || dd.detector == nil {
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

	if dd == nil || dd.detector == nil {
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
