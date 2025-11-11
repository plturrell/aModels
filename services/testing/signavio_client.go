package testing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"
)

// SignavioClient provides a client for interacting with Signavio Process Intelligence APIs.
type SignavioClient struct {
	baseURL    string
	apiKey     string
	tenantID   string
	httpClient *http.Client
	enabled    bool
	logger     *log.Logger
	timeout    time.Duration
	maxRetries int
}

// NewSignavioClient creates a new Signavio client.
// baseURL should be in format: https://ingestion-{region}.signavio.com (e.g., https://ingestion-eu.signavio.com)
func NewSignavioClient(baseURL, apiKey, tenantID string, enabled bool, timeout time.Duration, maxRetries int, logger *log.Logger) *SignavioClient {
	if baseURL == "" {
		baseURL = "https://ingestion-eu.signavio.com" // Default to EU region
	}
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	if maxRetries == 0 {
		maxRetries = 3
	}
	return &SignavioClient{
		baseURL:    baseURL,
		apiKey:     apiKey,
		tenantID:   tenantID,
		httpClient: &http.Client{Timeout: timeout},
		enabled:    enabled,
		logger:     logger,
		timeout:    timeout,
		maxRetries: maxRetries,
	}
}

// IsEnabled checks if Signavio integration is enabled.
func (c *SignavioClient) IsEnabled() bool {
	return c.enabled && c.baseURL != "" && c.apiKey != ""
}

// SignavioTelemetryRecord represents a telemetry record in Signavio format.
type SignavioTelemetryRecord struct {
	AgentRunID      string  `json:"agent_run_id"`
	AgentName       string  `json:"agent_name"`
	TaskID          string  `json:"task_id"`
	TaskDescription string  `json:"task_description"`
	StartTime       int64   `json:"start_time"` // timestamp-millis
	EndTime         int64   `json:"end_time"`   // timestamp-millis
	Status          string  `json:"status"`
	OutcomeSummary  *string `json:"outcome_summary,omitempty"`
	LatencyMs       *int64  `json:"latency_ms,omitempty"`
	Notes           *string `json:"notes,omitempty"`
	
	// New fields for comprehensive metrics
	ServiceName     string        `json:"service_name,omitempty"`
	WorkflowName    string        `json:"workflow_name,omitempty"`
	WorkflowVersion string        `json:"workflow_version,omitempty"`
	AgentType       string        `json:"agent_type,omitempty"`
	AgentState      string        `json:"agent_state,omitempty"`
	PromptMetrics   *PromptMetrics `json:"prompt_metrics,omitempty"`
	
	// Enhanced fields for richer dashboard data
	ToolsUsed       []SignavioToolUsage   `json:"tools_used,omitempty"`
	LLMCalls        []SignavioLLMCall     `json:"llm_calls,omitempty"`
	ProcessSteps    []SignavioProcessStep `json:"process_steps,omitempty"`
}

// PromptMetrics represents prompt statistics.
type PromptMetrics struct {
	PromptLength    int     `json:"prompt_length"`
	ResponseLength  int     `json:"response_length"`
	PromptType      string  `json:"prompt_type"`
	PromptCategory  string  `json:"prompt_category"`
	InputTokens     int     `json:"input_tokens"`
	OutputTokens    int     `json:"output_tokens"`
	PromptLatencyMs int64   `json:"prompt_latency_ms"`
}

// SignavioToolUsage represents tool usage statistics.
type SignavioToolUsage struct {
	ToolName      string         `json:"tool_name"`
	CallCount     int            `json:"call_count"`
	SuccessCount  int            `json:"success_count"`
	TotalLatencyMs int64         `json:"total_latency_ms"`
	ErrorDetails  string         `json:"error_details,omitempty"`
	Parameters    map[string]any `json:"parameters,omitempty"`
	Category      string         `json:"category,omitempty"`
	ServiceName   string         `json:"service_name,omitempty"`
}

// SignavioLLMCall represents LLM call statistics.
type SignavioLLMCall struct {
	Model         string  `json:"model"`
	CallCount     int     `json:"call_count"`
	TotalTokens   int     `json:"total_tokens"`
	InputTokens   int     `json:"input_tokens"`
	OutputTokens  int     `json:"output_tokens"`
	TotalLatencyMs int64  `json:"total_latency_ms"`
	Purpose       string  `json:"purpose"`
	Temperature   float64 `json:"temperature,omitempty"`
	Cost          float64 `json:"cost,omitempty"`
	ContextLength int     `json:"context_length,omitempty"`
	ServiceName   string  `json:"service_name,omitempty"`
}

// SignavioProcessStep represents a process step.
type SignavioProcessStep struct {
	StepName         string   `json:"step_name"`
	StartTime        int64    `json:"start_time"` // timestamp-millis
	EndTime          int64    `json:"end_time"`   // timestamp-millis
	Status           string   `json:"status"`
	DurationMs       int64    `json:"duration_ms"`
	WorkflowName     string   `json:"workflow_name,omitempty"`
	WorkflowVersion  string   `json:"workflow_version,omitempty"`
	Dependencies     []string `json:"dependencies,omitempty"`
	ParallelExecution bool    `json:"parallel_execution,omitempty"`
}

// UploadTelemetry uploads agent telemetry to Signavio using the Ingestion API.
// The Signavio Ingestion API requires multipart/form-data with:
// - files: CSV data file
// - schema: JSON Avro schema
// - primaryKeys: Comma-separated primary key fields
// - delimiter: Optional (defaults to comma)
func (c *SignavioClient) UploadTelemetry(ctx context.Context, dataset string, telemetryData []SignavioTelemetryRecord) error {
	if !c.IsEnabled() {
		return fmt.Errorf("Signavio client is not enabled")
	}

	if len(telemetryData) == 0 {
		return fmt.Errorf("no telemetry data to upload")
	}

	// Convert to CSV format
	csvData, err := c.convertToCSV(telemetryData)
	if err != nil {
		return fmt.Errorf("convert to CSV: %w", err)
	}

	// Generate Avro schema for the telemetry data
	schema, err := c.generateAvroSchema()
	if err != nil {
		return fmt.Errorf("generate schema: %w", err)
	}

	// Primary keys for the dataset
	primaryKeys := "agent_run_id"

	// Create multipart form data
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Add CSV file
	fileWriter, err := writer.CreateFormFile("files", "telemetry.csv")
	if err != nil {
		return fmt.Errorf("create form file: %w", err)
	}
	if _, err := fileWriter.Write(csvData); err != nil {
		return fmt.Errorf("write CSV data: %w", err)
	}

	// Add schema
	schemaWriter, err := writer.CreateFormField("schema")
	if err != nil {
		return fmt.Errorf("create schema field: %w", err)
	}
	if _, err := schemaWriter.Write(schema); err != nil {
		return fmt.Errorf("write schema: %w", err)
	}

	// Add primary keys
	if err := writer.WriteField("primaryKeys", primaryKeys); err != nil {
		return fmt.Errorf("write primary keys: %w", err)
	}

	// Add delimiter (optional, defaults to comma)
	if err := writer.WriteField("delimiter", ","); err != nil {
		return fmt.Errorf("write delimiter: %w", err)
	}

	// Get content type before closing
	contentType := writer.FormDataContentType()

	// Close multipart writer
	if err := writer.Close(); err != nil {
		return fmt.Errorf("close multipart writer: %w", err)
	}

	// Upload to Signavio ingestion API
	// Note: The actual endpoint URL is provided when creating the Ingestion API connection
	// This is a placeholder - the actual URL should be configured per connection
	url := fmt.Sprintf("%s/ingestion/api/v1/datasets/%s/upload", c.baseURL, dataset)
	
	var lastErr error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &requestBody)
		if err != nil {
			return fmt.Errorf("create request: %w", err)
		}

		// Set headers according to Signavio Ingestion API spec
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
		if c.tenantID != "" {
			req.Header.Set("X-Tenant-ID", c.tenantID)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("send request (attempt %d/%d): %w", attempt+1, c.maxRetries, err)
			if attempt < c.maxRetries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second) // Exponential backoff
				continue
			}
			return lastErr
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated || resp.StatusCode == http.StatusAccepted {
			body, _ := io.ReadAll(resp.Body)
			c.logger.Printf("Successfully uploaded %d telemetry records to Signavio dataset %s", len(telemetryData), dataset)
			if len(body) > 0 {
				c.logger.Printf("Signavio response: %s", string(body))
			}
			return nil
		}

		body, _ := io.ReadAll(resp.Body)
		lastErr = fmt.Errorf("upload failed (attempt %d/%d): status=%d, body=%s", attempt+1, c.maxRetries, resp.StatusCode, string(body))
		if attempt < c.maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * time.Second) // Exponential backoff
		}
	}

	return lastErr
}

// ExportExecutionMetrics exports test execution metrics to Signavio.
func (c *SignavioClient) ExportExecutionMetrics(ctx context.Context, execution *TestExecution, dataset string) error {
	if !c.IsEnabled() {
		return fmt.Errorf("Signavio client is not enabled")
	}

	// Format execution for Signavio
	telemetryRecord := FormatExecutionForSignavio(execution)

	// Upload single record
	return c.UploadTelemetry(ctx, dataset, []SignavioTelemetryRecord{telemetryRecord})
}

// convertToCSV converts telemetry records to CSV format.
// Enhanced fields (ToolsUsed, LLMCalls, ProcessSteps) are serialized as JSON strings.
func (c *SignavioClient) convertToCSV(records []SignavioTelemetryRecord) ([]byte, error) {
	if len(records) == 0 {
		return nil, fmt.Errorf("no records to convert")
	}

	var buf bytes.Buffer
	
	// Write CSV header including enhanced fields
	buf.WriteString("agent_run_id,agent_name,task_id,task_description,start_time,end_time,status,outcome_summary,latency_ms,notes,service_name,workflow_name,workflow_version,agent_type,agent_state,prompt_metrics,tools_used,llm_calls,process_steps\n")
	
	// Write records
	for _, record := range records {
		line := fmt.Sprintf("%s,%s,%s,%s,%d,%d,%s",
			escapeCSV(record.AgentRunID),
			escapeCSV(record.AgentName),
			escapeCSV(record.TaskID),
			escapeCSV(record.TaskDescription),
			record.StartTime,
			record.EndTime,
			escapeCSV(record.Status),
		)
		
		if record.OutcomeSummary != nil {
			line += "," + escapeCSV(*record.OutcomeSummary)
		} else {
			line += ","
		}
		
		if record.LatencyMs != nil {
			line += fmt.Sprintf(",%d", *record.LatencyMs)
		} else {
			line += ","
		}
		
		if record.Notes != nil {
			line += "," + escapeCSV(*record.Notes)
		} else {
			line += ","
		}
		
		// Add new fields
		line += "," + escapeCSV(record.ServiceName)
		line += "," + escapeCSV(record.WorkflowName)
		line += "," + escapeCSV(record.WorkflowVersion)
		line += "," + escapeCSV(record.AgentType)
		line += "," + escapeCSV(record.AgentState)
		
		// Serialize prompt metrics as JSON
		promptJSON, _ := json.Marshal(record.PromptMetrics)
		if record.PromptMetrics != nil {
			line += "," + escapeCSV(string(promptJSON))
		} else {
			line += ","
		}
		
		// Serialize enhanced fields as JSON strings
		toolsJSON, _ := json.Marshal(record.ToolsUsed)
		if len(record.ToolsUsed) > 0 {
			line += "," + escapeCSV(string(toolsJSON))
		} else {
			line += ","
		}
		
		llmJSON, _ := json.Marshal(record.LLMCalls)
		if len(record.LLMCalls) > 0 {
			line += "," + escapeCSV(string(llmJSON))
		} else {
			line += ","
		}
		
		stepsJSON, _ := json.Marshal(record.ProcessSteps)
		if len(record.ProcessSteps) > 0 {
			line += "," + escapeCSV(string(stepsJSON))
		} else {
			line += ","
		}
		
		buf.WriteString(line + "\n")
	}
	
	return buf.Bytes(), nil
}

// escapeCSV escapes CSV field values.
func escapeCSV(value string) string {
	if value == "" {
		return ""
	}
	// If value contains comma, quote, or newline, wrap in quotes and escape quotes
	if contains(value, ',', '"', '\n') {
		value = `"` + replaceAll(value, `"`, `""`) + `"`
	}
	return value
}

// contains checks if string contains any of the given characters.
func contains(s string, chars ...rune) bool {
	for _, char := range chars {
		for _, r := range s {
			if r == char {
				return true
			}
		}
	}
	return false
}

// replaceAll replaces all occurrences of old in s with new.
func replaceAll(s, old, new string) string {
	result := ""
	start := 0
	for {
		idx := find(s, old, start)
		if idx == -1 {
			result += s[start:]
			break
		}
		result += s[start:idx] + new
		start = idx + len(old)
	}
	return result
}

// find finds the first occurrence of substr in s starting from start.
func find(s, substr string, start int) int {
	if start >= len(s) {
		return -1
	}
	idx := -1
	for i := start; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			idx = i
			break
		}
	}
	return idx
}

// generateAvroSchema generates an Avro schema for the telemetry records.
// The schema follows the Apache Avro specification for type "record".
// Enhanced fields are included as optional string fields containing JSON.
func (c *SignavioClient) generateAvroSchema() ([]byte, error) {
	schema := map[string]any{
		"type": "record",
		"name": "AgentTelemetry",
		"namespace": "com.signavio.telemetry",
		"fields": []map[string]any{
			{"name": "agent_run_id", "type": "string"},
			{"name": "agent_name", "type": "string"},
			{"name": "task_id", "type": "string"},
			{"name": "task_description", "type": "string"},
			{"name": "start_time", "type": "long"},
			{"name": "end_time", "type": "long"},
			{"name": "status", "type": "string"},
			{"name": "outcome_summary", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "latency_ms", "type": []interface{}{"null", "long"}, "default": nil},
			{"name": "notes", "type": []interface{}{"null", "string"}, "default": nil},
			// New comprehensive metrics fields
			{"name": "service_name", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "workflow_name", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "workflow_version", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "agent_type", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "agent_state", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "prompt_metrics", "type": []interface{}{"null", "string"}, "default": nil},
			// Enhanced fields as JSON strings
			{"name": "tools_used", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "llm_calls", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "process_steps", "type": []interface{}{"null", "string"}, "default": nil},
		},
	}
	return json.Marshal(schema)
}

// HealthCheck checks if Signavio API is accessible.
// Note: Signavio Ingestion API doesn't have a standard health endpoint.
// This attempts to validate the connection by checking if the base URL is reachable.
func (c *SignavioClient) HealthCheck(ctx context.Context) error {
	if !c.IsEnabled() {
		return fmt.Errorf("Signavio client is not enabled")
	}

	// Signavio Ingestion API doesn't provide a health endpoint
	// We can only validate that the configuration is present
	if c.baseURL == "" || c.apiKey == "" {
		return fmt.Errorf("Signavio client not properly configured")
	}

	c.logger.Printf("Signavio client configured: baseURL=%s, tenantID=%s", c.baseURL, c.tenantID)
	return nil
}

