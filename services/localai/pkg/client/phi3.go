// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

package client

import (
	"context"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/types"
)

// Phi3Config holds model-specific defaults for phi-3.5-mini.
type Phi3Config struct {
	ModelName          string
	ContextWindow      int
	MaxTokens          int
	OptimalTemperature float64
}

// Phi3Integration wires SearchEnhancedLocalAI with Phi 3.5 Mini defaults.
type Phi3Integration struct {
	Client    *SearchEnhancedLocalAI
	ModelConf *Phi3Config
}

// NewPhi3Integration builds a Phi3Integration with the given endpoint/API key and context provider.
func NewPhi3Integration(endpoint, apiKey string, cp types.ContextProvider) *Phi3Integration {
	cfg := &Phi3Config{ModelName: "phi-3.5-mini", ContextWindow: 4096, MaxTokens: 1024, OptimalTemperature: 0.7}
	client := NewSearchEnhancedLocalAI(endpoint, cfg.ModelName, apiKey, cp)
	return &Phi3Integration{Client: client, ModelConf: cfg}
}

// Phi3OptimizedQuery applies light prompt shaping then calls GenerateWithContext.
func (pi *Phi3Integration) Phi3OptimizedQuery(ctx context.Context, query string, taskType string) (*types.GenerateResponse, error) {
	prompt := pi.optimizePromptForPhi3(query, taskType)
	req := &types.GenerateRequest{Prompt: prompt, Temperature: pi.ModelConf.OptimalTemperature, MaxTokens: pi.ModelConf.MaxTokens, TopP: 0.9}
	return pi.Client.GenerateWithContext(ctx, prompt, query, req)
}

func (pi *Phi3Integration) optimizePromptForPhi3(query string, taskType string) string {
	// Minimal specialization — can be extended per taskType
	switch taskType {
	case "reasoning":
		return "You are a careful reasoning assistant. Think step by step.\n\n" + query
	default:
		return query
	}
}

// SearchEnhancedLocalAI provides enhanced LocalAI functionality with context
type SearchEnhancedLocalAI struct {
	*LocalAIClient
	contextProvider types.ContextProvider
}

// NewSearchEnhancedLocalAI creates a new enhanced LocalAI client
func NewSearchEnhancedLocalAI(endpoint, modelName, apiKey string, cp types.ContextProvider) *SearchEnhancedLocalAI {
	return &SearchEnhancedLocalAI{
		LocalAIClient:   NewLocalAIClient(endpoint, modelName, apiKey),
		contextProvider: cp,
	}
}

// GenerateWithContext generates text with enhanced context
func (se *SearchEnhancedLocalAI) GenerateWithContext(ctx context.Context, prompt, query string, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	// Get context from provider if available
	if se.contextProvider != nil {
		context, err := se.contextProvider.GetContext(ctx, &types.SimpleTask{
			Prompt: prompt,
			Type:   "search_enhanced",
		})
		if err == nil && context != nil {
			// Enhance prompt with context
			enhancedPrompt := se.enhancePromptWithContext(prompt, context)
			req.Prompt = enhancedPrompt
		}
	}

	return se.Generate(ctx, req)
}

func (se *SearchEnhancedLocalAI) enhancePromptWithContext(prompt string, context map[string]interface{}) string {
	// Simple context enhancement - can be made more sophisticated
	enhanced := prompt
	if contextInfo, ok := context["context"].(string); ok {
		enhanced = "Context: " + contextInfo + "\n\n" + prompt
	}
	return enhanced
}
