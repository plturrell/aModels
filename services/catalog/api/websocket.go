package api

import (
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"github.com/plturrell/aModels/services/catalog/streaming"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins in development
	},
}

// WebSocketHandler provides WebSocket support for real-time updates.
type WebSocketHandler struct {
	eventStream *streaming.EventStream
	logger      *log.Logger
}

// NewWebSocketHandler creates a new WebSocket handler.
func NewWebSocketHandler(eventStream *streaming.EventStream, logger *log.Logger) *WebSocketHandler {
	return &WebSocketHandler{
		eventStream: eventStream,
		logger:      logger,
	}
}

// HandleWebSocket handles WebSocket connections.
func (h *WebSocketHandler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	h.logger.Printf("WebSocket connection established from %s", r.RemoteAddr)

	// Subscribe to events
	ctx := r.Context()
	eventsChan, err := h.eventStream.Subscribe(ctx, "websocket", r.RemoteAddr)
	if err != nil {
		h.logger.Printf("Failed to subscribe to events: %v", err)
		return
	}

	// Send ping ticker
	pingTicker := time.NewTicker(30 * time.Second)
	defer pingTicker.Stop()

	// Handle messages
	go func() {
		for {
			_, message, err := conn.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					h.logger.Printf("WebSocket error: %v", err)
				}
				return
			}
			// Handle client messages (e.g., subscribe/unsubscribe to specific event types)
			h.logger.Printf("Received message: %s", string(message))
		}
	}()

	// Send events to client
	for {
		select {
		case event, ok := <-eventsChan:
			if !ok {
				return
			}

			// Send event to WebSocket client
			if err := conn.WriteJSON(event); err != nil {
				h.logger.Printf("Failed to send event: %v", err)
				return
			}

		case <-pingTicker.C:
			// Send ping to keep connection alive
			if err := conn.WriteMessage(websocket.PingMessage, []byte{}); err != nil {
				return
			}

		case <-ctx.Done():
			return
		}
	}
}

// HandleWebSocketSubscribe handles WebSocket subscriptions for specific event types.
func (h *WebSocketHandler) HandleWebSocketSubscribe(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, "WebSocket upgrade failed", http.StatusBadRequest)
		return
	}
	defer conn.Close()

	// Parse subscription request
	var subscribeReq struct {
		EventTypes []string `json:"event_types"`
	}

	if err := conn.ReadJSON(&subscribeReq); err != nil {
		h.logger.Printf("Failed to read subscription request: %v", err)
		return
	}

	h.logger.Printf("Client subscribed to: %v", subscribeReq.EventTypes)

	// Subscribe to events
	ctx := r.Context()
	eventsChan, err := h.eventStream.Subscribe(ctx, "websocket", r.RemoteAddr)
	if err != nil {
		h.logger.Printf("Failed to subscribe: %v", err)
		return
	}

	// Filter and send events
	eventTypeMap := make(map[string]bool)
	for _, eventType := range subscribeReq.EventTypes {
		eventTypeMap[eventType] = true
	}

	for {
		select {
		case event, ok := <-eventsChan:
			if !ok {
				return
			}

			// Filter by event type
			if len(eventTypeMap) > 0 && !eventTypeMap[string(event.Type)] {
				continue
			}

			// Send event
			if err := conn.WriteJSON(event); err != nil {
				h.logger.Printf("Failed to send event: %v", err)
				return
			}

		case <-ctx.Done():
			return
		}
	}
}

// SendEvent sends an event to WebSocket clients.
func (h *WebSocketHandler) SendEvent(event streaming.Event) error {
	// Events are automatically forwarded via event stream subscription
	// This is a placeholder for direct sending if needed
	return nil
}

