package agents

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"
)

// AgentMarketplace provides a catalog of available agents.
type AgentMarketplace struct {
	db     *sql.DB
	logger *log.Logger
}

// NewAgentMarketplace creates a new agent marketplace.
func NewAgentMarketplace(db *sql.DB, logger *log.Logger) *AgentMarketplace {
	return &AgentMarketplace{
		db:     db,
		logger: logger,
	}
}

// AgentListing represents an agent in the marketplace.
type AgentListing struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Version     string                 `json:"version"`
	Author      string                 `json:"author"`
	Tags        []string               `json:"tags"`
	Rating      float64                `json:"rating"`
	UsageCount  int                    `json:"usage_count"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// ListAgents lists all available agents in the marketplace.
func (am *AgentMarketplace) ListAgents(ctx context.Context, filters map[string]interface{}) ([]*AgentListing, error) {
	// In production, would query database
	// For now, return default agents
	agents := []*AgentListing{
		{
			ID:          "agent-data-ingestion",
			Name:        "Data Ingestion Agent",
			Type:        "data_ingestion",
			Description: "Autonomous data ingestion from various sources",
			Version:     "1.0.0",
			Author:      "aModels Team",
			Tags:        []string{"ingestion", "autonomous", "data"},
			Rating:      4.5,
			UsageCount:  100,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		{
			ID:          "agent-mapping-rule",
			Name:        "Mapping Rule Agent",
			Type:        "mapping_rule",
			Description: "Automatic mapping rule learning and updates",
			Version:     "1.0.0",
			Author:      "aModels Team",
			Tags:        []string{"mapping", "learning", "automation"},
			Rating:      4.7,
			UsageCount:  150,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		{
			ID:          "agent-anomaly-detection",
			Name:        "Anomaly Detection Agent",
			Type:        "anomaly_detection",
			Description: "Automatic anomaly detection in data streams",
			Version:     "1.0.0",
			Author:      "aModels Team",
			Tags:        []string{"anomaly", "detection", "monitoring"},
			Rating:      4.6,
			UsageCount:  120,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		{
			ID:          "agent-test-generation",
			Name:        "Test Generation Agent",
			Type:        "test_generation",
			Description: "Automatic test scenario generation",
			Version:     "1.0.0",
			Author:      "aModels Team",
			Tags:        []string{"testing", "generation", "automation"},
			Rating:      4.4,
			UsageCount:  80,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
	}

	// Apply filters
	if tag, ok := filters["tag"].(string); ok {
		filtered := []*AgentListing{}
		for _, agent := range agents {
			for _, agentTag := range agent.Tags {
				if agentTag == tag {
					filtered = append(filtered, agent)
					break
				}
			}
		}
		agents = filtered
	}

	if agentType, ok := filters["type"].(string); ok {
		filtered := []*AgentListing{}
		for _, agent := range agents {
			if agent.Type == agentType {
				filtered = append(filtered, agent)
			}
		}
		agents = filtered
	}

	return agents, nil
}

// GetAgent retrieves an agent by ID.
func (am *AgentMarketplace) GetAgent(ctx context.Context, agentID string) (*AgentListing, error) {
	agents, err := am.ListAgents(ctx, nil)
	if err != nil {
		return nil, err
	}

	for _, agent := range agents {
		if agent.ID == agentID {
			return agent, nil
		}
	}

	return nil, fmt.Errorf("agent not found: %s", agentID)
}

// RegisterAgent registers a new agent in the marketplace.
func (am *AgentMarketplace) RegisterAgent(ctx context.Context, listing *AgentListing) error {
	if listing.ID == "" {
		return fmt.Errorf("agent ID is required")
	}

	if listing.CreatedAt.IsZero() {
		listing.CreatedAt = time.Now()
	}
	listing.UpdatedAt = time.Now()

	// In production, would store in database
	if am.logger != nil {
		am.logger.Printf("Agent registered in marketplace: %s (%s)", listing.ID, listing.Name)
	}

	return nil
}

// UpdateAgentUsage updates the usage count for an agent.
func (am *AgentMarketplace) UpdateAgentUsage(ctx context.Context, agentID string) error {
	// In production, would increment usage count in database
	if am.logger != nil {
		am.logger.Printf("Agent usage updated: %s", agentID)
	}

	return nil
}

