package rest

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/plturrell/aModels/services/runtime/orchestrator"
)

const (
	// Time allowed to write a message to the peer
	writeWait = 10 * time.Second

	// Time allowed to read the next pong message from the peer
	pongWait = 60 * time.Second

	// Send pings to peer with this period (must be less than pongWait)
	pingPeriod = (pongWait * 9) / 10

	// Maximum message size allowed from peer
	maxMessageSize = 512 * 1024
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		// Allow all origins in development; restrict in production
		return true
	},
}

// Client represents a WebSocket connection
type Client struct {
	orchestrator *orchestrator.Orchestrator
	conn         *websocket.Conn
	send         chan []byte
	hub          *Hub
	logger       *log.Logger
}

// Hub maintains the set of active clients and broadcasts messages to them
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

// NewHub creates a new Hub
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

// Run starts the hub
func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()

		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// Broadcast sends a message to all connected clients
func (h *Hub) Broadcast(data interface{}) error {
	message, err := json.Marshal(data)
	if err != nil {
		return err
	}
	select {
	case h.broadcast <- message:
	default:
		// Channel full, skip broadcast
	}
	return nil
}

// readPump pumps messages from the WebSocket connection to the hub
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.logger.Printf("websocket error: %v", err)
			}
			break
		}

		// Handle incoming messages (e.g., subscription requests)
		var msg map[string]interface{}
		if err := json.Unmarshal(message, &msg); err == nil {
			if msgType, ok := msg["type"].(string); ok {
				switch msgType {
				case "subscribe":
					// Send initial data
					c.sendDashboardUpdate()
				case "ping":
					// Respond to ping
					c.sendMessage(map[string]string{"type": "pong"})
				}
			}
		}
	}
}

// writePump pumps messages from the hub to the WebSocket connection
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued messages to the current websocket message
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// sendDashboardUpdate sends the current dashboard data to the client
func (c *Client) sendDashboardUpdate() {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	stats, templates, err := c.orchestrator.FetchDashboardData(ctx)
	if err != nil {
		c.logger.Printf("error fetching dashboard data: %v", err)
		return
	}

	update := map[string]interface{}{
		"type":      "dashboard_update",
		"timestamp": time.Now().Unix(),
		"stats":     stats,
		"templates": templates,
	}

	c.sendMessage(update)
}

// sendMessage sends a message to the client
func (c *Client) sendMessage(data interface{}) {
	message, err := json.Marshal(data)
	if err != nil {
		c.logger.Printf("error marshaling message: %v", err)
		return
	}

	select {
	case c.send <- message:
	default:
		c.logger.Printf("client send buffer full, dropping message")
	}
}

// ServeWebSocket handles WebSocket requests
func (h *Handler) ServeWebSocket(w http.ResponseWriter, r *http.Request, hub *Hub) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("websocket upgrade error: %v", err)
		return
	}

	client := &Client{
		orchestrator: h.orchestrator,
		conn:         conn,
		send:         make(chan []byte, 256),
		hub:          hub,
		logger:       log.New(log.Writer(), "websocket: ", log.LstdFlags),
	}

	client.hub.register <- client

	// Send initial dashboard data
	go client.sendDashboardUpdate()

	// Start client pumps
	go client.writePump()
	go client.readPump()
}
