package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

// EventStream provides event streaming capabilities using Redis Streams.
type EventStream struct {
	client *redis.Client
	logger *log.Logger
}

// NewEventStream creates a new event stream.
func NewEventStream(redisURL string, logger *log.Logger) (*EventStream, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	client := redis.NewClient(opt)

	// Test connection
	ctx := context.Background()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &EventStream{
		client: client,
		logger: logger,
	}, nil
}

// EventType represents the type of event.
type EventType string

const (
	EventTypeDataElementCreated   EventType = "data_element.created"
	EventTypeDataElementUpdated   EventType = "data_element.updated"
	EventTypeDataElementDeleted   EventType = "data_element.deleted"
	EventTypeQualityMetricsUpdated EventType = "quality.metrics_updated"
	EventTypeResearchCompleted    EventType = "research.completed"
	EventTypeDataProductCreated   EventType = "data_product.created"
)

// Event represents a catalog event.
type Event struct {
	Type      EventType              `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Source    string                 `json:"source"`
	Data      map[string]interface{} `json:"data"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Publish publishes an event to the stream.
func (es *EventStream) Publish(ctx context.Context, event Event) error {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	// Convert event to map for Redis Streams
	values := map[string]interface{}{
		"type":      string(event.Type),
		"timestamp": event.Timestamp.Format(time.RFC3339),
		"source":    event.Source,
	}

	// Add data fields
	dataJSON, _ := json.Marshal(event.Data)
	values["data"] = string(dataJSON)

	// Add metadata if present
	if len(event.Metadata) > 0 {
		metadataJSON, _ := json.Marshal(event.Metadata)
		values["metadata"] = string(metadataJSON)
	}

	// Publish to Redis Stream
	streamName := "catalog:events"
	_, err := es.client.XAdd(ctx, &redis.XAddArgs{
		Stream: streamName,
		Values: values,
	}).Result()

	if err != nil {
		return fmt.Errorf("failed to publish event: %w", err)
	}

	if es.logger != nil {
		es.logger.Printf("Published event: %s", event.Type)
	}

	return nil
}

// Subscribe subscribes to events from the stream.
func (es *EventStream) Subscribe(ctx context.Context, consumerGroup string, consumerName string) (<-chan Event, error) {
	streamName := "catalog:events"
	eventsChan := make(chan Event, 100)

	// Create consumer group if it doesn't exist
	err := es.client.XGroupCreateMkStream(ctx, streamName, consumerGroup, "0").Err()
	if err != nil && err.Error() != "BUSYGROUP Consumer Group name already exists" {
		return nil, fmt.Errorf("failed to create consumer group: %w", err)
	}

	// Start reading in a goroutine
	go func() {
		defer close(eventsChan)

		for {
			select {
			case <-ctx.Done():
				return
			default:
				// Read from stream
				streams, err := es.client.XReadGroup(ctx, &redis.XReadGroupArgs{
					Group:    consumerGroup,
					Consumer: consumerName,
					Streams:  []string{streamName, ">"},
					Count:    10,
					Block:    time.Second,
				}).Result()

				if err != nil {
					if err == redis.Nil {
						continue
					}
					if es.logger != nil {
						es.logger.Printf("Error reading stream: %v", err)
					}
					time.Sleep(time.Second)
					continue
				}

				// Process streams
				for _, stream := range streams {
					for _, message := range stream.Messages {
						event := es.parseEvent(message.Values)
						if event != nil {
							select {
							case eventsChan <- *event:
							case <-ctx.Done():
								return
							}
						}

						// Acknowledge message
						es.client.XAck(ctx, streamName, consumerGroup, message.ID)
					}
				}
			}
		}
	}()

	return eventsChan, nil
}

// parseEvent parses a Redis Stream message into an Event.
func (es *EventStream) parseEvent(values map[string]interface{}) *Event {
	eventTypeStr, ok := values["type"].(string)
	if !ok {
		return nil
	}

	timestampStr, ok := values["timestamp"].(string)
	if !ok {
		return nil
	}

	timestamp, err := time.Parse(time.RFC3339, timestampStr)
	if err != nil {
		return nil
	}

	source, _ := values["source"].(string)

	var data map[string]interface{}
	if dataStr, ok := values["data"].(string); ok {
		json.Unmarshal([]byte(dataStr), &data)
	}

	var metadata map[string]interface{}
	if metadataStr, ok := values["metadata"].(string); ok {
		json.Unmarshal([]byte(metadataStr), &metadata)
	}

	return &Event{
		Type:      EventType(eventTypeStr),
		Timestamp: timestamp,
		Source:    source,
		Data:      data,
		Metadata:  metadata,
	}
}

// Close closes the event stream connection.
func (es *EventStream) Close() error {
	return es.client.Close()
}

