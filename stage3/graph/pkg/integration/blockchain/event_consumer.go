//go:build hana

package blockchain

import (
	"context"
	"encoding/json"
	"time"

	"github.com/ethereum/go-ethereum/log"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
)

// BlockchainEvent represents a blockchain event from L1
type BlockchainEvent struct {
	ID              int64                  `json:"id"`
	EventType       string                 `json:"event_type"`
	TxHash          string                 `json:"tx_hash"`
	BlockNumber     int64                  `json:"block_number"`
	BlockHash       string                 `json:"block_hash"`
	AgentAddress    string                 `json:"agent_address"`
	ContractAddress string                 `json:"contract_address"`
	EventData       map[string]interface{} `json:"event_data"`
	Timestamp       time.Time              `json:"timestamp"`
	CreatedAt       time.Time              `json:"created_at"`
}

// EventHandler processes blockchain events
type EventHandler func(ctx context.Context, event *BlockchainEvent) error

// BlockchainEventConsumer consumes blockchain events and triggers Graph operations
type BlockchainEventConsumer struct {
	pool     *hanapool.Pool
	handlers map[string]EventHandler
}

// NewBlockchainEventConsumer creates a new blockchain event consumer
func NewBlockchainEventConsumer(pool *hanapool.Pool) *BlockchainEventConsumer {
	return &BlockchainEventConsumer{
		pool:     pool,
		handlers: make(map[string]EventHandler),
	}
}

// Subscribe registers an event handler for a specific event type
func (c *BlockchainEventConsumer) Subscribe(eventType string, handler EventHandler) {
	c.handlers[eventType] = handler
}

// Start begins consuming blockchain events
func (c *BlockchainEventConsumer) Start(ctx context.Context) error {
	log.Info("Starting blockchain event consumer")

	// Poll for new events
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	lastProcessedID := int64(0)

	for {
		select {
		case <-ctx.Done():
			log.Info("Blockchain event consumer stopping")
			return ctx.Err()
		case <-ticker.C:
			// Get new events since last processed
			events, err := c.getNewEvents(ctx, lastProcessedID)
			if err != nil {
				log.Error("Failed to get new events", "error", err)
				continue
			}

			// Process each event
			for _, event := range events {
				if err := c.processEvent(ctx, event); err != nil {
					log.Error("Failed to process event",
						"event_id", event.ID,
						"event_type", event.EventType,
						"error", err)
				}

				// Update last processed ID
				if event.ID > lastProcessedID {
					lastProcessedID = event.ID
				}
			}
		}
	}
}

// getNewEvents retrieves new blockchain events
func (c *BlockchainEventConsumer) getNewEvents(ctx context.Context, lastID int64) ([]*BlockchainEvent, error) {
	query := `
		SELECT id, event_type, tx_hash, block_number, block_hash, 
		       agent_address, contract_address, event_data, timestamp, created_at
		FROM BLOCKCHAIN_EVENTS 
		WHERE id > ? 
		ORDER BY id ASC 
		LIMIT 100
	`

	rows, err := c.pool.Query(ctx, query, lastID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []*BlockchainEvent
	for rows.Next() {
		event := &BlockchainEvent{}
		var eventDataJSON string

		err := rows.Scan(
			&event.ID,
			&event.EventType,
			&event.TxHash,
			&event.BlockNumber,
			&event.BlockHash,
			&event.AgentAddress,
			&event.ContractAddress,
			&eventDataJSON,
			&event.Timestamp,
			&event.CreatedAt,
		)
		if err != nil {
			return nil, err
		}

		// Parse event data JSON
		if eventDataJSON != "" {
			if err := json.Unmarshal([]byte(eventDataJSON), &event.EventData); err != nil {
				log.Warn("Failed to parse event data JSON", "event_id", event.ID, "error", err)
				event.EventData = make(map[string]interface{})
			}
		} else {
			event.EventData = make(map[string]interface{})
		}

		events = append(events, event)
	}

	return events, nil
}

// processEvent processes a single blockchain event
func (c *BlockchainEventConsumer) processEvent(ctx context.Context, event *BlockchainEvent) error {
	handler, exists := c.handlers[event.EventType]
	if !exists {
		// No handler for this event type, skip
		return nil
	}

	return handler(ctx, event)
}

// Graph-specific event handlers

// HandleAgentRegistrationEvent processes agent registration events
func HandleAgentRegistrationEvent(ctx context.Context, event *BlockchainEvent) error {
	log.Info("Processing agent registration event",
		"agent_address", event.AgentAddress,
		"tx_hash", event.TxHash,
		"block_number", event.BlockNumber)

	// Trigger Graph workflow for agent onboarding
	// This would integrate with the Graph runtime to start agent workflows
	// For now, we'll just log the event
	return nil
}

// HandleContractEvent processes smart contract events
func HandleContractEvent(ctx context.Context, event *BlockchainEvent) error {
	log.Info("Processing contract event",
		"contract_address", event.ContractAddress,
		"event_data", event.EventData,
		"tx_hash", event.TxHash)

	// Trigger Graph nodes based on contract events
	// This would integrate with Graph dispatchers
	return nil
}

// HandleBlockEvent processes block mining events
func HandleBlockEvent(ctx context.Context, event *BlockchainEvent) error {
	log.Debug("Processing block event",
		"block_number", event.BlockNumber,
		"block_hash", event.BlockHash,
		"tx_count", event.EventData["tx_count"])

	// Update Graph state with new block information
	// This could trigger periodic Graph maintenance tasks
	return nil
}

// HandleTransactionEvent processes transaction events
func HandleTransactionEvent(ctx context.Context, event *BlockchainEvent) error {
	log.Debug("Processing transaction event",
		"tx_hash", event.TxHash,
		"from", event.EventData["from"],
		"to", event.EventData["to"])

	// Process transaction for Graph workflows
	// This could trigger Graph nodes based on transaction patterns
	return nil
}

// Default event handler for unhandled event types
func DefaultEventHandler(ctx context.Context, event *BlockchainEvent) error {
	log.Debug("Processing default event",
		"event_type", event.EventType,
		"event_id", event.ID)

	return nil
}
