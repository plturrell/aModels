package telemetry

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	postgresv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/gen/v1"
)

// Config captures dial options for the telemetry client.
type Config struct {
	Address      string
	PrivacyLevel string
	UserIDHash   string
}

// FlowRunRecord describes the payload recorded for a Langflow execution.
type FlowRunRecord struct {
	OperationID  string
	Operation    string
	LibraryType  string
	FlowID       string
	LocalFlowID  string
	SessionID    string
	InputValue   string
	Inputs       map[string]any
	Tweaks       map[string]any
	Stream       bool
	Metadata     map[string]any
	Result       map[string]any
	Error        error
	ErrorMessage string
	Latency      time.Duration
	StartedAt    time.Time
	CompletedAt  time.Time
	PrivacyLevel string
	UserIDHash   string
}

// Client provides a light-weight wrapper around the PostgresLangService gRPC API.
type Client struct {
	conn           *grpc.ClientConn
	svc            postgresv1.PostgresLangServiceClient
	defaultPrivacy string
	defaultUser    string
}

// Dial establishes a gRPC client connection using the provided configuration.
func Dial(ctx context.Context, cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.Address) == "" {
		return nil, fmt.Errorf("telemetry address is required")
	}

	conn, err := grpc.DialContext(
		ctx,
		cfg.Address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("dial telemetry service: %w", err)
	}

	client := &Client{
		conn:           conn,
		svc:            postgresv1.NewPostgresLangServiceClient(conn),
		defaultPrivacy: cfg.PrivacyLevel,
		defaultUser:    cfg.UserIDHash,
	}
	return client, nil
}

// Close releases the underlying gRPC connection.
func (c *Client) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// LogFlowRun pushes a Langflow execution record to the Postgres telemetry service.
func (c *Client) LogFlowRun(ctx context.Context, record FlowRunRecord) error {
	if c == nil {
		return fmt.Errorf("telemetry client not initialised")
	}

	opID := strings.TrimSpace(record.OperationID)
	if opID == "" {
		opID = uuid.NewString()
	}

	library := record.LibraryType
	if library == "" {
		library = "langflow"
	}

	operation := record.Operation
	if operation == "" {
		operation = "run_flow"
	}

	start := record.StartedAt
	if start.IsZero() {
		start = time.Now()
	}
	completed := record.CompletedAt
	if completed.IsZero() {
		completed = start.Add(record.Latency)
	}

	inputPayload := map[string]any{
		"flow_id":       record.FlowID,
		"local_flow_id": record.LocalFlowID,
		"session_id":    record.SessionID,
		"input_value":   record.InputValue,
		"inputs":        safeMap(record.Inputs),
		"tweaks":        safeMap(record.Tweaks),
		"stream":        record.Stream,
	}
	if len(record.Metadata) > 0 {
		inputPayload["metadata"] = record.Metadata
	}

	outputPayload := safeMap(record.Result)

	inputStruct, err := structpb.NewStruct(inputPayload)
	if err != nil {
		return fmt.Errorf("encode telemetry input: %w", err)
	}

	outputStruct, err := structpb.NewStruct(outputPayload)
	if err != nil {
		return fmt.Errorf("encode telemetry output: %w", err)
	}

	status := postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS
	errorMessage := strings.TrimSpace(record.ErrorMessage)
	if errorMessage == "" && record.Error != nil {
		errorMessage = record.Error.Error()
	}
	if errorMessage != "" {
		status = postgresv1.OperationStatus_OPERATION_STATUS_ERROR
	}

	privacy := record.PrivacyLevel
	if privacy == "" {
		privacy = c.defaultPrivacy
	}

	userHash := record.UserIDHash
	if userHash == "" {
		userHash = c.defaultUser
	}

	op := &postgresv1.LangOperation{
		Id:           opID,
		LibraryType:  library,
		Operation:    operation,
		Input:        inputStruct,
		Output:       outputStruct,
		Status:       status,
		Error:        errorMessage,
		LatencyMs:    record.Latency.Milliseconds(),
		CreatedAt:    timestamppb.New(start),
		CompletedAt:  timestamppb.New(completed),
		SessionId:    record.SessionID,
		UserIdHash:   userHash,
		PrivacyLevel: privacy,
	}

	_, err = c.svc.LogLangOperation(ctx, &postgresv1.LogLangOperationRequest{Operation: op})
	if err != nil {
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
