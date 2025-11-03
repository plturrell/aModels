package grpcserver

import (
	"context"
	"fmt"
	"strings"

	localaiv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/api/localai/v1"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/client"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/types"
)

// Config configures the LocalAI gRPC bridge.
type Config struct {
	Endpoint    string
	Model       string
	Temperature float64
	MaxTokens   int32
	APIKey      string
}

// Server implements localai.v1.ChatServiceServer.
type Server struct {
	localaiv1.UnimplementedChatServiceServer

	client      *client.LocalAIClient
	temperature float64
	maxTokens   int32
}

// New creates a new gRPC bridge server instance.
func New(cfg Config) (*Server, error) {
	endpoint := strings.TrimRight(cfg.Endpoint, "/")
	if endpoint == "" {
		return nil, fmt.Errorf("endpoint must not be empty")
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		return nil, fmt.Errorf("model must not be empty")
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 0.7
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 512
	}

	c := client.NewLocalAIClient(endpoint, model, cfg.APIKey)
	return &Server{
		client:      c,
		temperature: cfg.Temperature,
		maxTokens:   cfg.MaxTokens,
	}, nil
}

// ChatCompletion forwards the request to the LocalAI HTTP endpoint and returns the result.
func (s *Server) ChatCompletion(ctx context.Context, req *localaiv1.ChatCompletionRequest) (*localaiv1.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("at least one message is required")
	}

	messages := make([]types.ChatMessage, 0, len(req.Messages))
	for _, msg := range req.Messages {
		role := strings.TrimSpace(msg.GetRole())
		if role == "" {
			role = "user"
		}
		messages = append(messages, types.ChatMessage{
			Role:    role,
			Content: msg.GetContent(),
		})
	}

	modelName := req.GetModel()
	if strings.TrimSpace(modelName) != "" {
		s.client.ModelName = modelName
	}

	genReq := &types.GenerateRequest{
		Temperature: s.temperature,
		MaxTokens:   int(req.GetMaxTokens()),
	}
	if req.GetTemperature() > 0 {
		genReq.Temperature = req.GetTemperature()
	}
	if req.GetMaxTokens() > 0 {
		genReq.MaxTokens = int(req.GetMaxTokens())
	} else {
		genReq.MaxTokens = int(s.maxTokens)
	}

	resp, err := s.client.GenerateChat(ctx, messages, genReq)
	if err != nil {
		return nil, fmt.Errorf("localai chat request failed: %w", err)
	}

	return &localaiv1.ChatCompletionResponse{
		Content:      resp.Text,
		TokensUsed:   int32(resp.TokensUsed),
		FinishReason: resp.FinishReason,
	}, nil
}
