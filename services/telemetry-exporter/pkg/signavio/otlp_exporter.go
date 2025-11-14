package signavio

import (
	"context"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"go.opentelemetry.io/otel/sdk/trace"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	resourcepb "go.opentelemetry.io/proto/otlp/resource/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	
	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/file"
	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/llm"
	"github.com/plturrell/aModels/services/testing"
)

// SignavioExporter exports OTLP traces to Signavio Process Intelligence API.
type SignavioExporter struct {
	client      *testing.SignavioClient
	dataset     string
	batchSize   int
	batchBuffer []*coltracepb.ExportTraceServiceRequest
	mu          sync.Mutex
	logger      *log.Logger
}

// SignavioExporterConfig configures the Signavio exporter.
type SignavioExporterConfig struct {
	BaseURL    string
	APIKey     string
	TenantID   string
	Dataset    string
	BatchSize  int
	Timeout    time.Duration
	MaxRetries int
	Logger     *log.Logger
}

// NewSignavioExporter creates a new Signavio exporter.
func NewSignavioExporter(cfg SignavioExporterConfig) (*SignavioExporter, error) {
	if cfg.Dataset == "" {
		return nil, fmt.Errorf("dataset is required")
	}
	if cfg.Logger == nil {
		cfg.Logger = log.New(io.Discard, "", 0)
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 100 // Default batch size
	}

	client := testing.NewSignavioClient(cfg.BaseURL, cfg.APIKey, cfg.TenantID, true, cfg.Timeout, cfg.MaxRetries, cfg.Logger)

	return &SignavioExporter{
		client:      client,
		dataset:     cfg.Dataset,
		batchSize:   cfg.BatchSize,
		batchBuffer: make([]*coltracepb.ExportTraceServiceRequest, 0, cfg.BatchSize),
		logger:      cfg.Logger,
	}, nil
}

// ExportTraces exports traces to Signavio.
func (se *SignavioExporter) ExportTraces(ctx context.Context, traces *coltracepb.ExportTraceServiceRequest) error {
	se.mu.Lock()
	se.batchBuffer = append(se.batchBuffer, traces)
	shouldFlush := len(se.batchBuffer) >= se.batchSize
	batch := se.batchBuffer
	if shouldFlush {
		se.batchBuffer = make([]*coltracepb.ExportTraceServiceRequest, 0, se.batchSize)
	}
	se.mu.Unlock()

	if shouldFlush {
		return se.flushBatch(ctx, batch)
	}
	return nil
}

// flushBatch flushes a batch of traces to Signavio.
func (se *SignavioExporter) flushBatch(ctx context.Context, batch []*coltracepb.ExportTraceServiceRequest) error {
	// Convert OTLP traces to Signavio format
	signavioRecords := make([]testing.SignavioTelemetryRecord, 0)
	for _, request := range batch {
		records := se.convertOTLPToSignavio(request)
		signavioRecords = append(signavioRecords, records...)
	}

	if len(signavioRecords) == 0 {
		return nil
	}

	// Upload to Signavio
	return se.client.UploadTelemetry(ctx, se.dataset, signavioRecords)
}

// Flush flushes any pending traces.
func (se *SignavioExporter) Flush(ctx context.Context) error {
	se.mu.Lock()
	batch := se.batchBuffer
	se.batchBuffer = make([]*coltracepb.ExportTraceServiceRequest, 0, se.batchSize)
	se.mu.Unlock()

	if len(batch) > 0 {
		return se.flushBatch(ctx, batch)
	}
	return nil
}

// Shutdown flushes pending traces and shuts down the exporter.
func (se *SignavioExporter) Shutdown(ctx context.Context) error {
	return se.Flush(ctx)
}

// ToSpanExporter converts SignavioExporter to trace.SpanExporter.
func (se *SignavioExporter) ToSpanExporter() trace.SpanExporter {
	return &signavioSpanExporter{exporter: se}
}

// signavioSpanExporter adapts SignavioExporter to trace.SpanExporter interface.
type signavioSpanExporter struct {
	exporter *SignavioExporter
}

// ExportSpans exports spans to Signavio.
func (sse *signavioSpanExporter) ExportSpans(ctx context.Context, spans []trace.ReadOnlySpan) error {
	// Convert spans to OTLP format (reuse conversion from file exporter)
	request := file.ConvertSpansToOTLP(spans)
	if request == nil {
		return nil // No spans to export
	}
	return sse.exporter.ExportTraces(ctx, request)
}

// Shutdown shuts down the exporter.
func (sse *signavioSpanExporter) Shutdown(ctx context.Context) error {
	return sse.exporter.Shutdown(ctx)
}


// convertOTLPToSignavio converts OTLP traces to Signavio telemetry records.
func (se *SignavioExporter) convertOTLPToSignavio(request *coltracepb.ExportTraceServiceRequest) []testing.SignavioTelemetryRecord {
	records := make([]testing.SignavioTelemetryRecord, 0)

	for _, resourceSpan := range request.ResourceSpans {
		for _, scopeSpan := range resourceSpan.ScopeSpans {
			for _, span := range scopeSpan.Spans {
				record := se.spanToSignavioRecord(span, resourceSpan.Resource)
				records = append(records, record)
			}
		}
	}

	return records
}

// spanToSignavioRecord converts an OTLP span to a Signavio telemetry record.
func (se *SignavioExporter) spanToSignavioRecord(span *tracepb.Span, resource *resourcepb.Resource) testing.SignavioTelemetryRecord {
	// Extract agent information from attributes
	agentName := extractAttribute(span.Attributes, "agent.name", "agent_name", "service.name")
	agentType := extractAttribute(span.Attributes, "agent.type", "agent_type", "agent_framework")
	agentRunID := extractAttribute(span.Attributes, "agent.run_id", "run_id", "trace_id")
	// Convert SpanId from []byte to hex string
	taskID := fmt.Sprintf("%x", span.SpanId)
	taskDescription := span.Name

	// Calculate timing
	startTime := time.Unix(0, int64(span.StartTimeUnixNano))
	endTime := time.Unix(0, int64(span.EndTimeUnixNano))
	duration := endTime.Sub(startTime)

	// Determine status
	status := "success"
	if span.Status != nil && span.Status.Code == tracepb.Status_STATUS_CODE_ERROR {
		status = "error"
	}

	// Extract outcome summary
	var outcomeSummary *string
	if span.Status != nil && span.Status.Message != "" {
		msg := span.Status.Message
		outcomeSummary = &msg
	}

	latencyMs := duration.Milliseconds()

	record := testing.SignavioTelemetryRecord{
		AgentRunID:      agentRunID,
		AgentName:       agentName,
		TaskID:          taskID,
		TaskDescription: taskDescription,
		StartTime:       startTime.UnixMilli(),
		EndTime:         endTime.UnixMilli(),
		Status:          status,
		OutcomeSummary:  outcomeSummary,
		LatencyMs:       &latencyMs,
		ServiceName:     extractAttribute(resource.Attributes, "service.name", "service_name"),
		AgentType:       agentType,
	}

	// Extract additional metadata from attributes
	record.WorkflowName = extractAttribute(span.Attributes, "workflow.name", "workflow_name")
	record.WorkflowVersion = extractAttribute(span.Attributes, "workflow.version", "workflow_version")
	record.AgentState = extractAttribute(span.Attributes, "agent.state", "state")

	// Extract tool usage from events
	record.ToolsUsed = extractToolUsage(span.Events)
	record.LLMCalls = extractLLMCalls(span.Events)
	record.ProcessSteps = extractProcessSteps(span.Events)

	// Extract prompt metrics if available
	if promptMetrics := extractPromptMetrics(span.Attributes); promptMetrics != nil {
		record.PromptMetrics = promptMetrics
	}

	// Extract LLM-specific information using OpenLLMetry conventions
	if llmInfo := llm.ExtractLLMInfo(span); llmInfo != nil && llmInfo.HasLLMAttributes {
		// Enhance LLM calls with OpenLLMetry attributes
		if len(record.LLMCalls) == 0 {
			// Create LLM call entry if none exists
			llmCall := testing.SignavioLLMCall{
				Model:   llmInfo.Model,
				Purpose: llmInfo.RequestType,
			}
			if llmInfo.PromptTokens > 0 {
				llmCall.InputTokens = int(llmInfo.PromptTokens)
			}
			if llmInfo.CompletionTokens > 0 {
				llmCall.OutputTokens = int(llmInfo.CompletionTokens)
			}
			if llmInfo.TotalTokens > 0 {
				llmCall.TotalTokens = int(llmInfo.TotalTokens)
			}
			record.LLMCalls = []testing.SignavioLLMCall{llmCall}
		} else {
			// Enhance existing LLM call entries
			for i := range record.LLMCalls {
				if record.LLMCalls[i].Model == "" && llmInfo.Model != "" {
					record.LLMCalls[i].Model = llmInfo.Model
				}
				if record.LLMCalls[i].InputTokens == 0 && llmInfo.PromptTokens > 0 {
					record.LLMCalls[i].InputTokens = int(llmInfo.PromptTokens)
				}
				if record.LLMCalls[i].OutputTokens == 0 && llmInfo.CompletionTokens > 0 {
					record.LLMCalls[i].OutputTokens = int(llmInfo.CompletionTokens)
				}
				if record.LLMCalls[i].TotalTokens == 0 && llmInfo.TotalTokens > 0 {
					record.LLMCalls[i].TotalTokens = int(llmInfo.TotalTokens)
				}
			}
		}
		
		// Store LLM system/provider in agent type if not already set
		if record.AgentType == "" && llmInfo.System != "" {
			record.AgentType = "llm:" + llmInfo.System
		}
	}

	return record
}

// Helper functions for attribute extraction

func extractAttribute(attrs []*commonpb.KeyValue, keys ...string) string {
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

func extractToolUsage(events []*tracepb.Span_Event) []testing.SignavioToolUsage {
	toolMap := make(map[string]*testing.SignavioToolUsage)
	
	for _, event := range events {
		if event.Name == "tool.call" || event.Name == "tool.result" {
			toolName := extractAttribute(event.Attributes, "tool.name", "tool_name")
			if toolName == "" {
				continue
			}
			
			tool, exists := toolMap[toolName]
			if !exists {
				tool = &testing.SignavioToolUsage{
					ToolName: toolName,
				}
				toolMap[toolName] = tool
			}
			
			if event.Name == "tool.call" {
				tool.CallCount++
			} else if event.Name == "tool.result" {
				if extractAttribute(event.Attributes, "error") == "" {
					tool.SuccessCount++
				} else {
					tool.ErrorDetails = extractAttribute(event.Attributes, "error", "error_message")
				}
			}
			
			// Extract latency if available
			if latencyStr := extractAttribute(event.Attributes, "latency_ms", "duration_ms"); latencyStr != "" {
				// Parse and accumulate
			}
		}
	}
	
	result := make([]testing.SignavioToolUsage, 0, len(toolMap))
	for _, tool := range toolMap {
		result = append(result, *tool)
	}
	return result
}

func extractLLMCalls(events []*tracepb.Span_Event) []testing.SignavioLLMCall {
	// Aggregate LLM calls by model and purpose
	callMap := make(map[string]*testing.SignavioLLMCall)
	
	for _, event := range events {
		if event.Name == "llm.call" || event.Name == "llm.response" {
			model := extractAttribute(event.Attributes, "model", "model_id")
			purpose := extractAttribute(event.Attributes, "purpose", "llm.purpose")
			if purpose == "" {
				purpose = "inference"
			}
			
			key := model + ":" + purpose
			call, exists := callMap[key]
			if !exists {
				call = &testing.SignavioLLMCall{
					Model:   model,
					Purpose: purpose,
				}
				callMap[key] = call
			}
			
			call.CallCount++
			if event.Name == "llm.call" {
				call.InputTokens += parseIntAttribute(event.Attributes, "input_tokens", "prompt_tokens")
			} else if event.Name == "llm.response" {
				call.OutputTokens += parseIntAttribute(event.Attributes, "output_tokens", "completion_tokens")
				call.TotalLatencyMs += int64(parseIntAttribute(event.Attributes, "latency_ms", "duration_ms"))
			}
			call.TotalTokens = call.InputTokens + call.OutputTokens
		}
	}
	
	// Convert map to slice
	calls := make([]testing.SignavioLLMCall, 0, len(callMap))
	for _, call := range callMap {
		calls = append(calls, *call)
	}
	
	return calls
}

func extractProcessSteps(events []*tracepb.Span_Event) []testing.SignavioProcessStep {
	steps := make([]testing.SignavioProcessStep, 0)
	stepMap := make(map[string]*testing.SignavioProcessStep) // Track steps by name
	
	for _, event := range events {
		if event.Name == "process.step" || event.Name == "workflow.step" {
			stepName := extractAttribute(event.Attributes, "step.name", "step_name", "name")
			if stepName == "" {
				continue
			}
			
			step, exists := stepMap[stepName]
			if !exists {
				step = &testing.SignavioProcessStep{
					StepName: stepName,
					Status:   extractAttribute(event.Attributes, "status", "step_status", "state"),
				}
				stepMap[stepName] = step
			}
			
			eventTime := time.Unix(0, int64(event.TimeUnixNano)).UnixMilli()
			if step.StartTime == 0 || eventTime < step.StartTime {
				step.StartTime = eventTime
			}
			if eventTime > step.EndTime {
				step.EndTime = eventTime
			}
			step.DurationMs = step.EndTime - step.StartTime
		}
	}
	
	// Convert map to slice
	for _, step := range stepMap {
		steps = append(steps, *step)
	}
	
	return steps
}

func extractPromptMetrics(attrs []*commonpb.KeyValue) *testing.PromptMetrics {
	hasMetrics := false
	metrics := &testing.PromptMetrics{}
	
	if promptType := extractAttribute(attrs, "prompt.type", "prompt_type"); promptType != "" {
		metrics.PromptType = promptType
		hasMetrics = true
	}
	if promptCategory := extractAttribute(attrs, "prompt.category", "prompt_category"); promptCategory != "" {
		metrics.PromptCategory = promptCategory
		hasMetrics = true
	}
	if inputTokens := parseIntAttribute(attrs, "prompt.input_tokens", "input_tokens"); inputTokens > 0 {
		metrics.InputTokens = inputTokens
		hasMetrics = true
	}
	if outputTokens := parseIntAttribute(attrs, "prompt.output_tokens", "output_tokens"); outputTokens > 0 {
		metrics.OutputTokens = outputTokens
		hasMetrics = true
	}
	if latencyMs := parseIntAttribute(attrs, "prompt.latency_ms", "prompt_latency_ms"); latencyMs > 0 {
		metrics.PromptLatencyMs = int64(latencyMs)
		hasMetrics = true
	}
	
	if hasMetrics {
		return metrics
	}
	return nil
}

func parseIntAttribute(attrs []*commonpb.KeyValue, keys ...string) int {
	for _, key := range keys {
		for _, attr := range attrs {
			if attr.Key == key && attr.Value != nil {
				if intVal := attr.Value.GetIntValue(); intVal != 0 {
					return int(intVal)
				}
			}
		}
	}
	return 0
}

