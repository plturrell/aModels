package llm

import (
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
)

// LLM semantic convention attribute keys (based on OpenLLMetry)
// These follow the OpenTelemetry GenAI semantic conventions
const (
	// LLM System/Provider
	AttrLLMSystem = "gen_ai.system"
	AttrLLMRequestType = "llm.request.type"
	AttrLLMRequestModel = "gen_ai.request.model"
	AttrLLMRequestModelName = "llm.request.model_name"
	
	// Token Usage
	AttrLLMUsageTotalTokens = "llm.usage.total_tokens"
	AttrLLMUsagePromptTokens = "gen_ai.usage.prompt_tokens"
	AttrLLMUsageCompletionTokens = "gen_ai.usage.completion_tokens"
	AttrLLMUsageTokenType = "llm.usage.token_type"
	
	// Cache Usage
	AttrLLMUsageCacheCreationInputTokens = "gen_ai.usage.cache_creation_input_tokens"
	AttrLLMUsageCacheReadInputTokens = "gen_ai.usage.cache_read_input_tokens"
	
	// Request Parameters
	AttrLLMTemperature = "gen_ai.request.temperature"
	AttrLLMTopP = "gen_ai.request.top_p"
	AttrLLMTopK = "llm.top_k"
	AttrLLMMaxTokens = "gen_ai.request.max_tokens"
	AttrLLMIsStreaming = "llm.is_streaming"
	AttrLLMFrequencyPenalty = "llm.frequency_penalty"
	AttrLLMPresencePenalty = "llm.presence_penalty"
	AttrLLMStopSequences = "llm.chat.stop_sequences"
	
	// Response
	AttrLLMResponseFinishReason = "llm.response.finish_reason"
	AttrLLMResponseStopReason = "llm.response.stop_reason"
	
	// Cost (if available)
	AttrLLMCost = "gen_ai.usage.cost"
	
	// Content
	AttrLLMContentCompletionChunk = "llm.content.completion.chunk"
	
	// Legacy/Alternative attribute names for compatibility
	AttrModel = "model"
	AttrModelID = "model_id"
	AttrInputTokens = "input_tokens"
	AttrOutputTokens = "output_tokens"
	AttrPromptTokens = "prompt_tokens"
	AttrCompletionTokens = "completion_tokens"
)

// LLMTraceInfo extracts LLM-specific information from a span
type LLMTraceInfo struct {
	System            string
	Model             string
	RequestType       string
	TotalTokens       int64
	PromptTokens      int64
	CompletionTokens  int64
	CacheCreationTokens int64
	CacheReadTokens   int64
	Temperature       float64
	TopP              float64
	TopK              int64
	MaxTokens         int64
	IsStreaming       bool
	FinishReason      string
	StopReason        string
	Cost              float64
	HasLLMAttributes  bool
}

// ExtractLLMInfo extracts LLM-specific information from span attributes
func ExtractLLMInfo(span *tracepb.Span) *LLMTraceInfo {
	info := &LLMTraceInfo{}
	
	// Check if this span has LLM attributes
	info.System = extractStringAttribute(span.Attributes, AttrLLMSystem, "llm.system", "provider")
	info.Model = extractStringAttribute(span.Attributes, AttrLLMRequestModel, AttrLLMRequestModelName, AttrModel, AttrModelID)
	info.RequestType = extractStringAttribute(span.Attributes, AttrLLMRequestType, "llm.type", "request_type")
	
	// Token usage
	info.TotalTokens = extractIntAttribute(span.Attributes, AttrLLMUsageTotalTokens, "total_tokens")
	info.PromptTokens = extractIntAttribute(span.Attributes, AttrLLMUsagePromptTokens, AttrInputTokens, AttrPromptTokens)
	info.CompletionTokens = extractIntAttribute(span.Attributes, AttrLLMUsageCompletionTokens, AttrOutputTokens, AttrCompletionTokens)
	
	// Cache usage
	info.CacheCreationTokens = extractIntAttribute(span.Attributes, AttrLLMUsageCacheCreationInputTokens)
	info.CacheReadTokens = extractIntAttribute(span.Attributes, AttrLLMUsageCacheReadInputTokens)
	
	// Request parameters
	info.Temperature = extractFloatAttribute(span.Attributes, AttrLLMTemperature, "temperature")
	info.TopP = extractFloatAttribute(span.Attributes, AttrLLMTopP, "top_p")
	info.TopK = extractIntAttribute(span.Attributes, AttrLLMTopK, "top_k")
	info.MaxTokens = extractIntAttribute(span.Attributes, AttrLLMMaxTokens, "max_tokens")
	info.IsStreaming = extractBoolAttribute(span.Attributes, AttrLLMIsStreaming, "is_streaming", "streaming")
	
	// Response
	info.FinishReason = extractStringAttribute(span.Attributes, AttrLLMResponseFinishReason, "finish_reason")
	info.StopReason = extractStringAttribute(span.Attributes, AttrLLMResponseStopReason, "stop_reason")
	
	// Cost
	info.Cost = extractFloatAttribute(span.Attributes, AttrLLMCost, "cost", "usage.cost")
	
	// Determine if this is an LLM span
	info.HasLLMAttributes = info.System != "" || info.Model != "" || info.RequestType != "" || info.TotalTokens > 0
	
	return info
}

// IsLLMSpan checks if a span represents an LLM operation
func IsLLMSpan(span *tracepb.Span) bool {
	info := ExtractLLMInfo(span)
	return info.HasLLMAttributes
}

// Helper functions for attribute extraction

func extractStringAttribute(attrs []*commonpb.KeyValue, keys ...string) string {
	for _, key := range keys {
		for _, attr := range attrs {
			if attr.Key == key && attr.Value != nil {
				if strVal := attr.Value.GetStringValue(); strVal != "" {
					return strVal
				}
			}
		}
	}
	return ""
}

func extractIntAttribute(attrs []*commonpb.KeyValue, keys ...string) int64 {
	for _, key := range keys {
		for _, attr := range attrs {
			if attr.Key == key && attr.Value != nil {
				if intVal := attr.Value.GetIntValue(); intVal != 0 {
					return intVal
				}
			}
		}
	}
	return 0
}

func extractFloatAttribute(attrs []*commonpb.KeyValue, keys ...string) float64 {
	for _, key := range keys {
		for _, attr := range attrs {
			if attr.Key == key && attr.Value != nil {
				if doubleVal := attr.Value.GetDoubleValue(); doubleVal != 0 {
					return doubleVal
				}
			}
		}
	}
	return 0
}

func extractBoolAttribute(attrs []*commonpb.KeyValue, keys ...string) bool {
	for _, key := range keys {
		for _, attr := range attrs {
			if attr.Key == key && attr.Value != nil {
				if boolVal := attr.Value.GetBoolValue() {
					return boolVal
				}
			}
		}
	}
	return false
}

