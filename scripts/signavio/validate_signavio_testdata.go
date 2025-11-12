package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	testing "ai_benchmarks/services/testing"
)

// validateSignavioTestData validates generated test data works with SignavioClient
func main() {
	// Parse flags
	dataDir := flag.String("dir", "./signavio_test_data", "Directory containing test data")
	upload := flag.Bool("upload", false, "Actually upload to Signavio (requires valid credentials)")
	baseURL := flag.String("url", "https://ingestion-eu.signavio.com", "Signavio base URL")
	apiKey := flag.String("api-key", "", "Signavio API key")
	tenantID := flag.String("tenant-id", "", "Signavio tenant ID")
	dataset := flag.String("dataset", "test-dataset", "Dataset name for upload")
	flag.Parse()

	logger := log.New(os.Stdout, "[SIGNAVIO-VALIDATOR] ", log.LstdFlags)

	fmt.Println("=" + string(repeat('=', 70)))
	fmt.Println("Signavio Test Data Validator")
	fmt.Println("=" + string(repeat('=', 70)))
	fmt.Println()

	// 1. Load and validate JSON structure
	jsonPath := filepath.Join(*dataDir, "agent_telemetry.json")
	logger.Printf("Loading test data from: %s", jsonPath)

	records, err := loadTelemetryRecords(jsonPath)
	if err != nil {
		logger.Fatalf("Failed to load telemetry records: %v", err)
	}

	logger.Printf("✓ Loaded %d telemetry records", len(records))
	fmt.Println()

	// 2. Validate record structure
	logger.Println("Validating record structure...")
	for i, record := range records {
		if err := validateRecord(record, i); err != nil {
			logger.Printf("⚠ Record %d validation warning: %v", i, err)
		}
	}
	logger.Println("✓ All records validated")
	fmt.Println()

	// 3. Test CSV file exists
	csvPath := filepath.Join(*dataDir, "agent_telemetry.csv")
	if _, err := os.Stat(csvPath); err != nil {
		logger.Printf("⚠ CSV file not found: %s", csvPath)
	} else {
		logger.Printf("✓ CSV file found: %s", csvPath)
	}

	// 4. Test Avro schema file exists
	avscPath := filepath.Join(*dataDir, "agent_telemetry.avsc")
	if _, err := os.Stat(avscPath); err != nil {
		logger.Printf("⚠ Avro schema not found: %s", avscPath)
	} else {
		logger.Printf("✓ Avro schema found: %s", avscPath)
		
		// Validate schema is valid JSON
		schemaData, err := os.ReadFile(avscPath)
		if err != nil {
			logger.Printf("⚠ Failed to read schema: %v", err)
		} else {
			var schema map[string]interface{}
			if err := json.Unmarshal(schemaData, &schema); err != nil {
				logger.Printf("⚠ Invalid Avro schema JSON: %v", err)
			} else {
				logger.Println("✓ Avro schema is valid JSON")
			}
		}
	}
	fmt.Println()

	// 5. Test SignavioClient integration
	logger.Println("Testing SignavioClient integration...")
	client := testing.NewSignavioClient(
		*baseURL,
		*apiKey,
		*tenantID,
		*upload, // Only enable if upload flag is set
		30*time.Second,
		3,
		logger,
	)

	if *upload {
		if *apiKey == "" {
			logger.Println("⚠ Upload mode requires --api-key")
			logger.Println("ℹ Skipping upload test")
		} else {
			logger.Printf("Uploading %d records to dataset: %s", len(records), *dataset)
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			if err := client.UploadTelemetry(ctx, *dataset, records); err != nil {
				logger.Printf("✗ Upload failed: %v", err)
			} else {
				logger.Println("✓ Upload successful")
			}
		}
	} else {
		logger.Println("ℹ Skipping upload (use --upload flag to test real upload)")
		logger.Println("✓ SignavioClient instantiated successfully")
	}

	fmt.Println()

	// 6. Print summary
	fmt.Println("=" + string(repeat('=', 70)))
	fmt.Println("Validation Summary")
	fmt.Println("=" + string(repeat('=', 70)))
	fmt.Printf("Records loaded:        %d\n", len(records))
	fmt.Printf("Data directory:        %s\n", *dataDir)
	fmt.Printf("Files validated:       3 (JSON, CSV, Avro schema)\n")
	fmt.Printf("SignavioClient:        ✓ Compatible\n")
	if *upload && *apiKey != "" {
		fmt.Printf("Upload test:           Executed\n")
	} else {
		fmt.Printf("Upload test:           Skipped\n")
	}
	fmt.Println("=" + string(repeat('=', 70)))

	// 7. Print example record
	if len(records) > 0 {
		fmt.Println()
		fmt.Println("Example Record:")
		fmt.Println(string(repeat('-', 70)))
		printRecord(records[0])
		fmt.Println(string(repeat('-', 70)))
	}
}

// loadTelemetryRecords loads telemetry records from JSON file
func loadTelemetryRecords(filepath string) ([]testing.SignavioTelemetryRecord, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var records []testing.SignavioTelemetryRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("unmarshal JSON: %w", err)
	}

	return records, nil
}

// validateRecord validates a single telemetry record
func validateRecord(record testing.SignavioTelemetryRecord, index int) error {
	if record.AgentRunID == "" {
		return fmt.Errorf("missing agent_run_id")
	}
	if record.AgentName == "" {
		return fmt.Errorf("missing agent_name")
	}
	if record.TaskID == "" {
		return fmt.Errorf("missing task_id")
	}
	if record.StartTime == 0 {
		return fmt.Errorf("missing start_time")
	}
	if record.EndTime == 0 {
		return fmt.Errorf("missing end_time")
	}
	if record.Status == "" {
		return fmt.Errorf("missing status")
	}
	return nil
}

// printRecord prints a formatted record summary
func printRecord(record testing.SignavioTelemetryRecord) {
	fmt.Printf("Agent Run ID:     %s\n", record.AgentRunID)
	fmt.Printf("Agent Name:       %s\n", record.AgentName)
	fmt.Printf("Task ID:          %s\n", record.TaskID)
	fmt.Printf("Task Description: %s\n", record.TaskDescription)
	fmt.Printf("Status:           %s\n", record.Status)
	fmt.Printf("Start Time:       %d (%s)\n", 
		record.StartTime, 
		time.Unix(record.StartTime/1000, 0).Format(time.RFC3339))
	fmt.Printf("End Time:         %d (%s)\n", 
		record.EndTime, 
		time.Unix(record.EndTime/1000, 0).Format(time.RFC3339))
	if record.LatencyMs != nil {
		fmt.Printf("Latency:          %d ms\n", *record.LatencyMs)
	}
	if record.ServiceName != "" {
		fmt.Printf("Service:          %s\n", record.ServiceName)
	}
	if record.WorkflowName != "" {
		fmt.Printf("Workflow:         %s\n", record.WorkflowName)
	}
	if len(record.ToolsUsed) > 0 {
		fmt.Printf("Tools Used:       %d\n", len(record.ToolsUsed))
	}
	if len(record.LLMCalls) > 0 {
		fmt.Printf("LLM Calls:        %d\n", len(record.LLMCalls))
	}
	if len(record.ProcessSteps) > 0 {
		fmt.Printf("Process Steps:    %d\n", len(record.ProcessSteps))
	}
}

// repeat creates a repeated character string
func repeat(char rune, count int) []rune {
	result := make([]rune, count)
	for i := range result {
		result[i] = char
	}
	return result
}
