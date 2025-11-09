// Package server interfaces defines the core interfaces for the server package.
// These interfaces provide abstraction layers for model providers, backend providers,
// request processors, and domain detectors, enabling better testability and
// modularity in the server implementation.
package server

import (
	"context"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/transformers"
)

// ModelProvider provides access to models of different types
type ModelProvider interface {
	// GetSafetensorsModel returns a safetensors model by key
	GetSafetensorsModel(key string) (*ai.VaultGemma, bool)
	// GetGGUFModel returns a GGUF model by key
	GetGGUFModel(key string) (*gguf.Model, bool)
	// GetTransformerClient returns a transformers client by key
	GetTransformerClient(key string) (*transformers.Client, bool)
	// HasModel checks if a model exists for the given key
	HasModel(key string) bool
}

// BackendProvider provides backend-specific operations
type BackendProvider interface {
	// GetBackendType returns the backend type for a domain
	GetBackendType(domain string) string
	// IsBackendAvailable checks if a backend is available
	IsBackendAvailable(backendType string) bool
	// GetBackendEndpoint returns the endpoint for a backend
	GetBackendEndpoint(backendType, domain string) string
}

// RequestProcessor processes chat completion requests
type RequestProcessor interface {
	// ProcessChatRequest processes a chat completion request
	ProcessChatRequest(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
	// ProcessStreamingRequest processes a streaming chat request
	ProcessStreamingRequest(ctx context.Context, req *ChatRequest, writer *StreamWriter) error
	// ProcessFunctionCallingRequest processes a function calling request
	ProcessFunctionCallingRequest(ctx context.Context, req *FunctionCallingRequest) (*FunctionCallingResponse, error)
}

// DomainDetector detects the appropriate domain for a prompt
type DomainDetector interface {
	// DetectDomain detects the domain for a given prompt
	DetectDomain(prompt string, availableDomains []string) string
	// GetDomainConfig returns the configuration for a domain
	GetDomainConfig(domain string) (*domain.DomainConfig, error)
	// GetDefaultDomain returns the default domain
	GetDefaultDomain() string
}

// ChatRequest represents a chat completion request
type ChatRequest struct {
	Model       string
	Messages   []ChatMessage
	MaxTokens  int
	Temperature float64
	TopP       float64
	TopK       int
	Domains    []string
	Images     []string
	VisionPrompt string
}

// ChatMessage represents a message in a chat request
type ChatMessage struct {
	Role    string
	Content string
}

// ChatResponse represents a chat completion response
type ChatResponse struct {
	ID       string
	Object   string
	Created  int64
	Model    string
	Choices  []ChatChoice
	Usage    TokenUsage
	Metadata map[string]interface{}
}

// ChatChoice represents a choice in a chat response
type ChatChoice struct {
	Index        int
	Message      ChatMessage
	FinishReason string
}

// TokenUsage represents token usage information
type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

