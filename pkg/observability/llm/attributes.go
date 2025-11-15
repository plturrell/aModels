package llm

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// OpenLLMetry semantic convention attribute keys
// These follow the OpenTelemetry GenAI semantic conventions used by OpenLLMetry
const (
	// LLM System/Provider
	AttrGenAISystem        = "gen_ai.system"
	AttrLLMRequestType     = "llm.request.type"
	AttrGenAIRequestModel   = "gen_ai.request.model"
	AttrLLMRequestModelName = "llm.request.model_name"

	// Token Usage
	AttrLLMUsageTotalTokens      = "llm.usage.total_tokens"
	AttrGenAIUsagePromptTokens   = "gen_ai.usage.prompt_tokens"
	AttrGenAIUsageCompletionTokens = "gen_ai.usage.completion_tokens"
	AttrLLMUsageTokenType        = "llm.usage.token_type"

	// Cache Usage
	AttrGenAIUsageCacheCreationInputTokens = "gen_ai.usage.cache_creation_input_tokens"
	AttrGenAIUsageCacheReadInputTokens     = "gen_ai.usage.cache_read_input_tokens"

	// Request Parameters
	AttrGenAIRequestTemperature = "gen_ai.request.temperature"
	AttrGenAIRequestTopP        = "gen_ai.request.top_p"
	AttrLLMTopK                 = "llm.top_k"
	AttrGenAIRequestMaxTokens   = "gen_ai.request.max_tokens"
	AttrLLMIsStreaming          = "llm.is_streaming"
	AttrLLMFrequencyPenalty     = "llm.frequency_penalty"
	AttrLLMPresencePenalty        = "llm.presence_penalty"
	AttrLLMChatStopSequences      = "llm.chat.stop_sequences"

	// Response
	AttrLLMResponseFinishReason = "llm.response.finish_reason"
	AttrLLMResponseStopReason   = "llm.response.stop_reason"

	// Cost (if available)
	AttrGenAIUsageCost = "gen_ai.usage.cost"
)

// LLMRequestConfig holds configuration for an LLM request
type LLMRequestConfig struct {
	System          string  // Provider/system name (e.g., "localai", "openai")
	Model           string  // Model identifier
	RequestType     string  // Request type: "chat", "completion", "embedding"
	Temperature     float64 // Temperature setting
	TopP            float64 // Top-p sampling parameter
	TopK            int64   // Top-k sampling parameter
	MaxTokens       int64   // Maximum tokens in response
	IsStreaming     bool    // Whether the request is streaming
	FrequencyPenalty float64 // Frequency penalty
	PresencePenalty  float64 // Presence penalty
}

// LLMResponseInfo holds information about an LLM response
type LLMResponseInfo struct {
	PromptTokens      int64   // Input/prompt tokens
	CompletionTokens  int64   // Output/completion tokens
	TotalTokens       int64   // Total tokens
	FinishReason      string  // Completion reason (e.g., "stop", "length")
	StopReason        string  // Stop reason if applicable
	Cost              float64 // Cost of the operation (if available)
	CacheCreationTokens int64 // Tokens used for cache creation
	CacheReadTokens     int64 // Tokens read from cache
}

// AddLLMRequestAttributes adds OpenLLMetry request attributes to a span
func AddLLMRequestAttributes(span trace.Span, config LLMRequestConfig) {
	if !span.IsRecording() {
		return
	}

	attrs := []attribute.KeyValue{}

	if config.System != "" {
		attrs = append(attrs, attribute.String(AttrGenAISystem, config.System))
	}
	if config.Model != "" {
		attrs = append(attrs, attribute.String(AttrGenAIRequestModel, config.Model))
	}
	if config.RequestType != "" {
		attrs = append(attrs, attribute.String(AttrLLMRequestType, config.RequestType))
	}
	if config.Temperature > 0 {
		attrs = append(attrs, attribute.Float64(AttrGenAIRequestTemperature, config.Temperature))
	}
	if config.TopP > 0 {
		attrs = append(attrs, attribute.Float64(AttrGenAIRequestTopP, config.TopP))
	}
	if config.TopK > 0 {
		attrs = append(attrs, attribute.Int64(AttrLLMTopK, config.TopK))
	}
	if config.MaxTokens > 0 {
		attrs = append(attrs, attribute.Int64(AttrGenAIRequestMaxTokens, config.MaxTokens))
	}
	attrs = append(attrs, attribute.Bool(AttrLLMIsStreaming, config.IsStreaming))
	if config.FrequencyPenalty != 0 {
		attrs = append(attrs, attribute.Float64(AttrLLMFrequencyPenalty, config.FrequencyPenalty))
	}
	if config.PresencePenalty != 0 {
		attrs = append(attrs, attribute.Float64(AttrLLMPresencePenalty, config.PresencePenalty))
	}

	if len(attrs) > 0 {
		span.SetAttributes(attrs...)
	}
}

// AddLLMResponseAttributes adds OpenLLMetry response attributes to a span
func AddLLMResponseAttributes(span trace.Span, info LLMResponseInfo) {
	if !span.IsRecording() {
		return
	}

	attrs := []attribute.KeyValue{}

	if info.PromptTokens > 0 {
		attrs = append(attrs, attribute.Int64(AttrGenAIUsagePromptTokens, info.PromptTokens))
	}
	if info.CompletionTokens > 0 {
		attrs = append(attrs, attribute.Int64(AttrGenAIUsageCompletionTokens, info.CompletionTokens))
	}
	if info.TotalTokens > 0 {
		attrs = append(attrs, attribute.Int64(AttrLLMUsageTotalTokens, info.TotalTokens))
	} else if info.PromptTokens > 0 || info.CompletionTokens > 0 {
		// Calculate total if not provided
		total := info.PromptTokens + info.CompletionTokens
		attrs = append(attrs, attribute.Int64(AttrLLMUsageTotalTokens, total))
	}
	if info.FinishReason != "" {
		attrs = append(attrs, attribute.String(AttrLLMResponseFinishReason, info.FinishReason))
	}
	if info.StopReason != "" {
		attrs = append(attrs, attribute.String(AttrLLMResponseStopReason, info.StopReason))
	}
	if info.Cost > 0 {
		attrs = append(attrs, attribute.Float64(AttrGenAIUsageCost, info.Cost))
	}
	if info.CacheCreationTokens > 0 {
		attrs = append(attrs, attribute.Int64(AttrGenAIUsageCacheCreationInputTokens, info.CacheCreationTokens))
	}
	if info.CacheReadTokens > 0 {
		attrs = append(attrs, attribute.Int64(AttrGenAIUsageCacheReadInputTokens, info.CacheReadTokens))
	}

	if len(attrs) > 0 {
		span.SetAttributes(attrs...)
	}
}

// AddLLMAttributes is a convenience function that adds both request and response attributes
func AddLLMAttributes(span trace.Span, config LLMRequestConfig, info LLMResponseInfo) {
	AddLLMRequestAttributes(span, config)
	AddLLMResponseAttributes(span, info)
}

// NewLLMRequestConfig creates a new LLMRequestConfig with defaults for LocalAI
func NewLLMRequestConfig(model string, requestType string) LLMRequestConfig {
	return LLMRequestConfig{
		System:      "localai",
		Model:       model,
		RequestType: requestType,
		Temperature: 0.7, // Default temperature
		IsStreaming: false,
	}
}

