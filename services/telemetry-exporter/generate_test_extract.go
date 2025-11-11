package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	exporter "github.com/plturrell/aModels/services/telemetry-exporter/internal/exporter"
	testing "github.com/plturrell/aModels/services/testing"
)

func main() {
	// Create a realistic test telemetry response
	testTelemetry := &exporter.ExtractServiceResponse{
		SessionID: "test-session-" + fmt.Sprintf("%d", time.Now().Unix()),
		Metrics: exporter.ExtractMetricsSummary{
			UserPromptCount:       3,
			ToolCallCount:         15,
			ToolCallStartedCount:  15,
			ToolSuccessCount:      14,
			ToolErrorCount:        1,
			ToolSuccessRate:       0.933,
			AverageToolLatencyMs:  245.5,
			ModelChangeCount:      2,
			LastUserPrompt:        "Extract data from customer table and generate summary report",
			ServiceName:           "extract-service",
			WorkflowName:          "data-extraction-workflow",
			AgentType:             "data-extraction-agent",
			PromptCount:           3,
			TotalPromptTokens:     1250,
			TotalResponseTokens:   890,
			AveragePromptLatencyMs: 1200.0,
		},
		Events: []exporter.ExtractMetricEvent{
			{
				Timestamp: time.Now().Add(-10 * time.Minute),
				SessionID: "test-session-123",
				Type:      "user_prompt",
				Payload: map[string]any{
					"prompt":         "Extract data from customer table and generate summary report",
					"tokens":         450.0,
					"prompt_type":    "task",
					"prompt_category": "data_processing",
				},
			},
			{
				Timestamp: time.Now().Add(-9 * time.Minute),
				SessionID: "test-session-123",
				Type:      "model_change",
				Payload: map[string]any{
					"model":          "gpt-4",
					"mode":           "chat",
					"input_tokens":   450.0,
					"output_tokens":  320.0,
					"temperature":    0.7,
					"context_length": 8192,
					"service_name":   "openai-service",
				},
			},
			{
				Timestamp: time.Now().Add(-8 * time.Minute),
				SessionID: "test-session-123",
				Type:      "tool_call_started",
				Payload: map[string]any{
					"tool_name":    "database_query",
					"request_id":   "req-001",
					"category":     "data-access",
					"service_name": "postgres-service",
					"parameters": map[string]any{
						"table": "customers",
						"limit": 1000,
					},
				},
			},
			{
				Timestamp: time.Now().Add(-7 * time.Minute),
				SessionID: "test-session-123",
				Type:      "tool_call_completed",
				Payload: map[string]any{
					"tool_name":    "database_query",
					"request_id":   "req-001",
					"success":      true,
					"latency_ms":   180.5,
					"category":     "data-access",
					"service_name": "postgres-service",
				},
			},
			{
				Timestamp: time.Now().Add(-6 * time.Minute),
				SessionID: "test-session-123",
				Type:      "tool_call_started",
				Payload: map[string]any{
					"tool_name":    "data_transformation",
					"request_id":   "req-002",
					"category":     "data-processing",
					"service_name": "transform-service",
					"parameters": map[string]any{
						"operation": "aggregate",
						"group_by":  "region",
					},
				},
			},
			{
				Timestamp: time.Now().Add(-5 * time.Minute),
				SessionID: "test-session-123",
				Type:      "tool_call_completed",
				Payload: map[string]any{
					"tool_name":    "data_transformation",
					"request_id":   "req-002",
					"success":      true,
					"latency_ms":   320.0,
					"category":     "data-processing",
					"service_name": "transform-service",
				},
			},
			{
				Timestamp: time.Now().Add(-4 * time.Minute),
				SessionID: "test-session-123",
				Type:      "model_change",
				Payload: map[string]any{
					"model":          "gpt-4",
					"mode":           "chat",
					"input_tokens":   800.0,
					"output_tokens":  570.0,
					"temperature":    0.7,
					"context_length": 8192,
					"service_name":   "openai-service",
					"purpose":        "report_generation",
				},
			},
			{
				Timestamp: time.Now().Add(-3 * time.Minute),
				SessionID: "test-session-123",
				Type:      "tool_call_started",
				Payload: map[string]any{
					"tool_name":    "file_writer",
					"request_id":   "req-003",
					"category":     "io",
					"service_name": "storage-service",
				},
			},
			{
				Timestamp: time.Now().Add(-2 * time.Minute),
				SessionID: "test-session-123",
				Type:      "tool_call_completed",
				Payload: map[string]any{
					"tool_name":     "file_writer",
					"request_id":    "req-003",
					"success":       false,
					"latency_ms":    150.0,
					"error":         "Permission denied",
					"error_details": "Cannot write to /output/reports directory",
					"category":      "io",
					"service_name":  "storage-service",
				},
			},
			{
				Timestamp: time.Now().Add(-1 * time.Minute),
				SessionID: "test-session-123",
				Type:      "user_prompt",
				Payload: map[string]any{
					"prompt":         "Retry the file write operation with different path",
					"tokens":         380.0,
					"prompt_type":    "instruction",
					"prompt_category": "error_recovery",
				},
			},
		},
	}

	// Format for Signavio
	record := exporter.FormatFromExtractService(testTelemetry, "extract-service")

	// Create Signavio client to generate CSV and schema
	signavioClient := testing.NewSignavioClient(
		"https://ingestion-eu.signavio.com",
		"test-api-key",
		"test-tenant-id",
		true,
		30*time.Second,
		3,
		nil,
	)

	// Use the client's internal methods via a test upload (we'll extract the data)
	// Actually, let's create CSV and schema manually using the same format
	csvData := generateCSVForSignavio([]testing.SignavioTelemetryRecord{*record})
	schemaData := generateAvroSchemaForSignavio()

	// Write CSV file
	csvFile := "signavio_telemetry_export.csv"
	if err := os.WriteFile(csvFile, csvData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing CSV file: %v\n", err)
		os.Exit(1)
	}

	// Write Avro schema file
	schemaFile := "signavio_telemetry_schema.json"
	if err := os.WriteFile(schemaFile, schemaData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing schema file: %v\n", err)
		os.Exit(1)
	}

	// Write JSON record for reference
	jsonData, _ := json.MarshalIndent(record, "", "  ")
	jsonFile := "signavio_telemetry_record.json"
	if err := os.WriteFile(jsonFile, jsonData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing JSON file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✓ Generated Signavio export files:\n")
	fmt.Printf("  - %s (CSV data)\n", csvFile)
	fmt.Printf("  - %s (Avro schema)\n", schemaFile)
	fmt.Printf("  - %s (JSON record for reference)\n", jsonFile)
	fmt.Printf("\n")
	fmt.Printf("Session ID: %s\n", record.AgentRunID)
	fmt.Printf("Agent Name: %s\n", record.AgentName)
	fmt.Printf("Service: %s\n", record.ServiceName)
	fmt.Printf("Workflow: %s\n", record.WorkflowName)
	fmt.Printf("Status: %s\n", record.Status)
	fmt.Printf("Tools Used: %d\n", len(record.ToolsUsed))
	fmt.Printf("LLM Calls: %d\n", len(record.LLMCalls))
	fmt.Printf("Process Steps: %d\n", len(record.ProcessSteps))
}

// generateCSVForSignavio generates CSV in the same format as SignavioClient
func generateCSVForSignavio(records []testing.SignavioTelemetryRecord) []byte {
	var buf bytes.Buffer

	// Write CSV header
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

	return buf.Bytes()
}

// escapeCSV escapes CSV field values
func escapeCSV(value string) string {
	if value == "" {
		return ""
	}
	// If value contains comma, quote, or newline, wrap in quotes and escape quotes
	needsQuoting := false
	for _, r := range value {
		if r == ',' || r == '"' || r == '\n' {
			needsQuoting = true
			break
		}
	}
	if needsQuoting {
		value = `"` + bytes.ReplaceAll([]byte(value), []byte(`"`), []byte(`""`)) + `"`
		return string(value)
	}
	return value
}

// generateAvroSchemaForSignavio generates Avro schema
func generateAvroSchemaForSignavio() []byte {
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
			{"name": "service_name", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "workflow_name", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "workflow_version", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "agent_type", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "agent_state", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "prompt_metrics", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "tools_used", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "llm_calls", "type": []interface{}{"null", "string"}, "default": nil},
			{"name": "process_steps", "type": []interface{}{"null", "string"}, "default": nil},
		},
	}
	data, _ := json.Marshal(schema)
	return data
}
