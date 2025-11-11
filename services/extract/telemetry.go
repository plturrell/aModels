package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	postgresv1 "github.com/plturrell/aModels/services/postgres/pkg/gen/v1"
)

type telemetryConfig struct {
	Address          string
	LibraryType      string
	DefaultOperation string
	PrivacyLevel     string
	UserIDHash       string
	DialTimeout      time.Duration
	CallTimeout      time.Duration
}

type telemetryClient struct {
	conn             *grpc.ClientConn
	svc              postgresv1.PostgresLangServiceClient
	libraryType      string
	defaultOperation string
	defaultPrivacy   string
	defaultUser      string
	callTimeout      time.Duration
}

type telemetryRecord struct {
	LibraryType  string
	Operation    string
	Input        map[string]any
	Output       map[string]any
	Error        error
	ErrorMessage string
	StartedAt    time.Time
	CompletedAt  time.Time
	Latency      time.Duration
	SessionID    string
	PrivacyLevel string
	UserIDHash   string
}

func newTelemetryClient(ctx context.Context, cfg telemetryConfig) (*telemetryClient, error) {
	addr := strings.TrimSpace(cfg.Address)
	if addr == "" {
		return nil, fmt.Errorf("telemetry address is required")
	}

	dialTimeout := cfg.DialTimeout
	if dialTimeout <= 0 {
		dialTimeout = defaultDialTimeout
	}
	dialCtx, cancel := context.WithTimeout(ctx, dialTimeout)
	defer cancel()

	conn, err := grpc.DialContext(
		dialCtx,
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("dial telemetry service: %w", err)
	}

	callTimeout := cfg.CallTimeout
	if callTimeout <= 0 {
		callTimeout = defaultCallTimeout
	}

	library := strings.TrimSpace(cfg.LibraryType)
	if library == "" {
		library = defaultTelemetryLibrary
	}

	operation := strings.TrimSpace(cfg.DefaultOperation)
	if operation == "" {
		operation = defaultTelemetryOperation
	}

	client := &telemetryClient{
		conn:             conn,
		svc:              postgresv1.NewPostgresLangServiceClient(conn),
		libraryType:      library,
		defaultOperation: operation,
		defaultPrivacy:   strings.TrimSpace(cfg.PrivacyLevel),
		defaultUser:      strings.TrimSpace(cfg.UserIDHash),
		callTimeout:      callTimeout,
	}
	return client, nil
}

func (c *telemetryClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *telemetryClient) Log(ctx context.Context, record telemetryRecord) error {
	if c == nil || c.svc == nil {
		return nil
	}

	library := strings.TrimSpace(record.LibraryType)
	if library == "" {
		library = c.libraryType
	}

	operation := strings.TrimSpace(record.Operation)
	if operation == "" {
		operation = c.defaultOperation
	}

	start := record.StartedAt
	if start.IsZero() {
		start = time.Now()
	}

	completed := record.CompletedAt
	if completed.IsZero() {
		if record.Latency > 0 {
			completed = start.Add(record.Latency)
		} else {
			completed = start
		}
	}

	latency := record.Latency
	if latency <= 0 {
		latency = completed.Sub(start)
		if latency < 0 {
			latency = 0
		}
	}

	inputStruct, err := structpb.NewStruct(safeMap(record.Input))
	if err != nil {
		return fmt.Errorf("encode telemetry input: %w", err)
	}

	outputStruct, err := structpb.NewStruct(safeMap(record.Output))
	if err != nil {
		return fmt.Errorf("encode telemetry output: %w", err)
	}

	errorMessage := strings.TrimSpace(record.ErrorMessage)
	if errorMessage == "" && record.Error != nil {
		errorMessage = record.Error.Error()
	}

	status := postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS
	if errorMessage != "" {
		status = postgresv1.OperationStatus_OPERATION_STATUS_ERROR
	}

	privacy := strings.TrimSpace(record.PrivacyLevel)
	if privacy == "" {
		privacy = c.defaultPrivacy
	}

	user := strings.TrimSpace(record.UserIDHash)
	if user == "" {
		user = c.defaultUser
	}

	callCtx, cancel := context.WithTimeout(ctx, c.callTimeout)
	defer cancel()

	req := &postgresv1.LogLangOperationRequest{
		Operation: &postgresv1.LangOperation{
			LibraryType:  library,
			Operation:    operation,
			Input:        inputStruct,
			Output:       outputStruct,
			Status:       status,
			Error:        errorMessage,
			LatencyMs:    latency.Milliseconds(),
			CreatedAt:    timestamppb.New(start),
			CompletedAt:  timestamppb.New(completed),
			SessionId:    strings.TrimSpace(record.SessionID),
			UserIdHash:   user,
			PrivacyLevel: privacy,
		},
	}

	if status != postgresv1.OperationStatus_OPERATION_STATUS_ERROR {
		req.Operation.Error = ""
	}

	if _, err := c.svc.LogLangOperation(callCtx, req); err != nil {
		return fmt.Errorf("log lang operation: %w", err)
	}
	return nil
}

func safeMap(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}

func telemetryInputFromRequest(req extractRequest) map[string]any {
	summary := map[string]any{
		"prompt_description":        strings.TrimSpace(req.PromptDescription),
		"model_id":                  strings.TrimSpace(req.ModelID),
		"documents_count":           len(req.Documents),
		"text_or_documents_present": len(req.TextOrDocumentsRaw) > 0,
		"text_or_documents_bytes":   len(req.TextOrDocumentsRaw),
		"examples_count":            len(req.Examples),
	}

	if doc := strings.TrimSpace(req.Document); doc != "" {
		summary["document_preview"] = previewString(doc, 200)
		summary["document_chars"] = len([]rune(doc))
		summary["document_hash"] = hashString(doc)
	}

	if len(req.Documents) > 0 {
		const previewLimit = 3
		const previewLen = 120

		previews := make([]string, 0, minInt(len(req.Documents), previewLimit))
		hashes := make([]string, 0, len(req.Documents))
		totalBytes := 0

		for i, doc := range req.Documents {
			totalBytes += len(doc)
			hash := hashString(doc)
			hashes = append(hashes, hash)
			if i < previewLimit {
				previews = append(previews, previewString(doc, previewLen))
			}
		}

		summary["documents_preview"] = previews
		summary["documents_hashes"] = hashes
		summary["documents_total_bytes"] = totalBytes
	}

	if len(req.Examples) > 0 {
		classSet := map[string]struct{}{}
		for _, example := range req.Examples {
			for _, extraction := range example.Extractions {
				class := strings.TrimSpace(extraction.ExtractionClass)
				if class != "" {
					classSet[class] = struct{}{}
				}
			}
		}
		if len(classSet) > 0 {
			classes := make([]string, 0, len(classSet))
			for class := range classSet {
				classes = append(classes, class)
			}
			sort.Strings(classes)
			summary["examples_classes"] = classes
		}
	}

	return summary
}

func telemetryOutputFromResponse(resp *extractResponse) map[string]any {
	if resp == nil {
		return map[string]any{}
	}

	return map[string]any{
		"entities":         flattenEntities(resp.Entities),
		"extractions":      summariseExtractions(resp.Extractions),
		"extraction_count": len(resp.Extractions),
	}
}

func summariseExtractions(extractions []extractionResult) []map[string]any {
	if len(extractions) == 0 {
		return []map[string]any{}
	}

	summaries := make([]map[string]any, 0, len(extractions))
	for _, extraction := range extractions {
		entry := map[string]any{
			"class": strings.TrimSpace(extraction.ExtractionClass),
		}
		if text := strings.TrimSpace(extraction.ExtractionText); text != "" {
			entry["text_preview"] = previewString(text, 160)
			entry["text_chars"] = len([]rune(text))
			entry["text_hash"] = hashString(text)
		}
		if extraction.Attributes != nil {
			entry["attributes"] = extraction.Attributes
		}
		if extraction.StartIndex != nil {
			entry["start_index"] = *extraction.StartIndex
		}
		if extraction.EndIndex != nil {
			entry["end_index"] = *extraction.EndIndex
		}
		summaries = append(summaries, entry)
	}
	return summaries
}

func previewString(value string, maxLen int) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	runes := []rune(trimmed)
	if len(runes) <= maxLen {
		return trimmed
	}
	return string(runes[:maxLen]) + "..."
}

func hashString(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(trimmed))
	return fmt.Sprintf("%x", sum[:])
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func safeEntities(entities map[string][]string) map[string][]string {
	if entities == nil {
		return map[string][]string{}
	}
	return entities
}

func flattenEntities(entities map[string][]string) map[string]any {
	if len(entities) == 0 {
		return map[string]any{}
	}
	flattened := make(map[string]any, len(entities))
	for key, vals := range entities {
		asAny := make([]any, 0, len(vals))
		for _, v := range vals {
			asAny = append(asAny, v)
		}
		flattened[key] = asAny
	}
	return flattened
}
