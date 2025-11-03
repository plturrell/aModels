// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

package types

import (
	"context"
	"time"
)

// GenerateRequest captures the knobs for text or chat generations.
type GenerateRequest struct {
	Prompt           string
	Temperature      float64
	MaxTokens        int
	TopP             float64
	FrequencyPenalty float64
	PresencePenalty  float64
	StopSequences    []string
	Metadata         map[string]interface{}
}

// GenerateResponse captures the canonical response returned by LocalAI integrations.
type GenerateResponse struct {
	Text         string
	TokensUsed   int
	FinishReason string
	Metadata     map[string]interface{}
}

// ChatMessage represents a single message in a chat-style exchange.
type ChatMessage struct {
	Role    string                 `json:"role"`
	Content string                 `json:"content"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
}

// LoadBalancingStrategy enumerates available model selection strategies.
type LoadBalancingStrategy string

const (
	StrategyRoundRobin    LoadBalancingStrategy = "round_robin"
	StrategyLeastLoaded   LoadBalancingStrategy = "least_loaded"
	StrategyLowestLatency LoadBalancingStrategy = "lowest_latency"
	StrategyRandom        LoadBalancingStrategy = "random"
)

// ModelSpecification describes an individual model option exposed by LocalAI.
type ModelSpecification struct {
	Name           string
	AverageLatency time.Duration
	MaxTokens      int
	ContextWindow  int
	Metadata       map[string]interface{}
}

// ContextProvider returns additional context to enrich generation prompts.
type ContextProvider interface {
	GetContext(ctx context.Context, task *SimpleTask) (map[string]interface{}, error)
}

// SimpleTask is passed to context providers to request additional information.
type SimpleTask struct {
	Prompt   string
	Type     string
	Metadata map[string]interface{}
}
