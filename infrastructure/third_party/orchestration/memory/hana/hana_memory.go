package hana

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

// HANAChatMessageHistory implements memory interface using HANA
type HANAChatMessageHistory struct {
	pool           *hanapool.Pool
	memoryStore    *storage.MemoryStore
	agentID        string
	sessionID      string
	conversationID int64
}

// NewHANAChatMessageHistory creates a new HANA-backed chat message history
func NewHANAChatMessageHistory(pool *hanapool.Pool, agentID, sessionID string) (*HANAChatMessageHistory, error) {
	memoryStore := storage.NewMemoryStore(pool)

	// Get or create conversation
	ctx := context.Background()
	conversation, err := memoryStore.GetConversationBySession(ctx, agentID, sessionID)
	if err != nil {
		// Create new conversation
		conversationID, err := memoryStore.CreateConversation(ctx, agentID, sessionID)
		if err != nil {
			return nil, fmt.Errorf("failed to create conversation: %w", err)
		}
		conversation = &storage.Conversation{ID: conversationID}
	}

	return &HANAChatMessageHistory{
		pool:           pool,
		memoryStore:    memoryStore,
		agentID:        agentID,
		sessionID:      sessionID,
		conversationID: conversation.ID,
	}, nil
}

// AddMessage adds a message to the conversation
func (h *HANAChatMessageHistory) AddMessage(ctx context.Context, message llms.ChatMessage) error {
	role := string(message.GetType())
	content := message.GetContent()

	// Convert metadata - simplified since ChatMessage interface doesn't expose AdditionalKwargs
	metadata := make(map[string]string)

	_, err := h.memoryStore.AddMessage(ctx, h.conversationID, role, content, metadata)
	if err != nil {
		return fmt.Errorf("failed to add message: %w", err)
	}

	return nil
}

// AddUserMessage adds a user message
func (h *HANAChatMessageHistory) AddUserMessage(ctx context.Context, message string) error {
	chatMessage := llms.HumanChatMessage{Content: message}
	return h.AddMessage(ctx, chatMessage)
}

// AddAIMessage adds an AI message
func (h *HANAChatMessageHistory) AddAIMessage(ctx context.Context, message string) error {
	chatMessage := llms.AIChatMessage{Content: message}
	return h.AddMessage(ctx, chatMessage)
}

// AddSystemMessage adds a system message
func (h *HANAChatMessageHistory) AddSystemMessage(ctx context.Context, message string) error {
	chatMessage := llms.SystemChatMessage{Content: message}
	return h.AddMessage(ctx, chatMessage)
}

// Messages returns all messages in the conversation
func (h *HANAChatMessageHistory) Messages(ctx context.Context) ([]llms.ChatMessage, error) {
	storageMessages, err := h.memoryStore.GetMessages(ctx, h.conversationID, 0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}

	messages := make([]llms.ChatMessage, 0, len(storageMessages))
	for _, msg := range storageMessages {
		var chatMsg llms.ChatMessage
		switch msg.Role {
		case "user":
			chatMsg = llms.HumanChatMessage{Content: msg.Content}
		case "assistant":
			chatMsg = llms.AIChatMessage{Content: msg.Content}
		case "system":
			chatMsg = llms.SystemChatMessage{Content: msg.Content}
		default:
			chatMsg = llms.GenericChatMessage{Content: msg.Content, Role: msg.Role}
		}
		messages = append(messages, chatMsg)
	}

	return messages, nil
}

// RecentMessages returns recent messages up to the specified count
func (h *HANAChatMessageHistory) RecentMessages(ctx context.Context, count int) ([]llms.ChatMessage, error) {
	storageMessages, err := h.memoryStore.GetRecentMessages(ctx, h.conversationID, count)
	if err != nil {
		return nil, fmt.Errorf("failed to get recent messages: %w", err)
	}

	messages := make([]llms.ChatMessage, 0, len(storageMessages))
	for _, msg := range storageMessages {
		var chatMsg llms.ChatMessage
		switch msg.Role {
		case "user":
			chatMsg = llms.HumanChatMessage{Content: msg.Content}
		case "assistant":
			chatMsg = llms.AIChatMessage{Content: msg.Content}
		case "system":
			chatMsg = llms.SystemChatMessage{Content: msg.Content}
		default:
			chatMsg = llms.GenericChatMessage{Content: msg.Content, Role: msg.Role}
		}
		messages = append(messages, chatMsg)
	}

	return messages, nil
}

// Clear clears all messages from the conversation
func (h *HANAChatMessageHistory) Clear(ctx context.Context) error {
	// Delete all messages for this conversation
	query := `DELETE FROM messages WHERE conversation_id = ?`
	_, err := h.pool.Execute(ctx, query, h.conversationID)
	if err != nil {
		return fmt.Errorf("failed to clear messages: %w", err)
	}

	return nil
}

// GetConversationID returns the conversation ID
func (h *HANAChatMessageHistory) GetConversationID() int64 {
	return h.conversationID
}

// GetAgentID returns the agent ID
func (h *HANAChatMessageHistory) GetAgentID() string {
	return h.agentID
}

// GetSessionID returns the session ID
func (h *HANAChatMessageHistory) GetSessionID() string {
	return h.sessionID
}

// GetConversationSummary returns a summary of the conversation
func (h *HANAChatMessageHistory) GetConversationSummary(ctx context.Context, days int) (map[string]interface{}, error) {
	return h.memoryStore.GetConversationSummary(ctx, h.agentID, days)
}

// HANAChatMessageHistoryManager manages multiple chat message histories
type HANAChatMessageHistoryManager struct {
	pool      *hanapool.Pool
	histories map[string]*HANAChatMessageHistory
}

// NewHANAChatMessageHistoryManager creates a new manager
func NewHANAChatMessageHistoryManager(pool *hanapool.Pool) *HANAChatMessageHistoryManager {
	return &HANAChatMessageHistoryManager{
		pool:      pool,
		histories: make(map[string]*HANAChatMessageHistory),
	}
}

// GetOrCreateHistory gets or creates a chat message history
func (m *HANAChatMessageHistoryManager) GetOrCreateHistory(agentID, sessionID string) (*HANAChatMessageHistory, error) {
	key := fmt.Sprintf("%s:%s", agentID, sessionID)

	if history, exists := m.histories[key]; exists {
		return history, nil
	}

	history, err := NewHANAChatMessageHistory(m.pool, agentID, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to create history: %w", err)
	}

	m.histories[key] = history
	return history, nil
}

// GetHistory gets an existing chat message history
func (m *HANAChatMessageHistoryManager) GetHistory(agentID, sessionID string) (*HANAChatMessageHistory, bool) {
	key := fmt.Sprintf("%s:%s", agentID, sessionID)
	history, exists := m.histories[key]
	return history, exists
}

// RemoveHistory removes a chat message history from memory
func (m *HANAChatMessageHistoryManager) RemoveHistory(agentID, sessionID string) {
	key := fmt.Sprintf("%s:%s", agentID, sessionID)
	delete(m.histories, key)
}

// ListHistories returns all active histories
func (m *HANAChatMessageHistoryManager) ListHistories() []string {
	keys := make([]string, 0, len(m.histories))
	for key := range m.histories {
		keys = append(keys, key)
	}
	return keys
}
