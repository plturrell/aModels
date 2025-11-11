package postgresflight

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const operationsTicket = "operations/logs"

// OperationRow captures a row returned by the Postgres Flight server.
type OperationRow struct {
	ID           string         `json:"id"`
	LibraryType  string         `json:"library_type"`
	Operation    string         `json:"operation"`
	Status       string         `json:"status"`
	Error        string         `json:"error,omitempty"`
	LatencyMs    int64          `json:"latency_ms"`
	CreatedAt    time.Time      `json:"created_at"`
	CompletedAt  *time.Time     `json:"completed_at,omitempty"`
	SessionID    string         `json:"session_id"`
	UserIDHash   string         `json:"user_id_hash"`
	PrivacyLevel string         `json:"privacy_level"`
	Input        map[string]any `json:"input,omitempty"`
	Output       map[string]any `json:"output,omitempty"`
}

// FetchOperations pulls operation rows from the Flight server.
func FetchOperations(ctx context.Context, addr string) ([]OperationRow, error) {
	if addr == "" {
		return nil, fmt.Errorf("flight address required")
	}
	client, err := flight.NewClientWithMiddlewareCtx(ctx, addr, nil, nil, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial flight endpoint: %w", err)
	}
	defer client.Close()

	desc := &flight.FlightDescriptor{Type: flight.DescriptorPATH, Path: []string{"operations", "logs"}}
	info, err := client.GetFlightInfo(ctx, desc)
	if err != nil {
		return nil, fmt.Errorf("get flight info: %w", err)
	}
	if len(info.Endpoint) == 0 {
		return nil, nil
	}

	alloc := memory.NewGoAllocator()
	var rows []OperationRow

	for _, endpoint := range info.Endpoint {
		ticket := endpoint.Ticket
		if ticket == nil {
			ticket = &flight.Ticket{Ticket: []byte(operationsTicket)}
		}
		stream, err := client.DoGet(ctx, ticket)
		if err != nil {
			return nil, fmt.Errorf("flight doget: %w", err)
		}
		reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(alloc))
		if err != nil {
			stream.CloseSend()
			return nil, fmt.Errorf("record reader: %w", err)
		}
		for reader.Next() {
			record := reader.Record()
			rows = append(rows, recordToOperations(record)...)
		}
		reader.Release()
		_ = stream.CloseSend()
	}
	return rows, nil
}

func recordToOperations(record arrow.Record) []OperationRow {
	if record == nil {
		return nil
	}
	idCol := record.Column(0).(*array.String)
	libCol := record.Column(1).(*array.String)
	opCol := record.Column(2).(*array.String)
	statusCol := record.Column(3).(*array.String)
	errCol := record.Column(4).(*array.String)
	latencyCol := record.Column(5).(*array.Int64)
	createdCol := record.Column(6).(*array.String)
	completedCol := record.Column(7).(*array.String)
	sessionCol := record.Column(8).(*array.String)
	userCol := record.Column(9).(*array.String)
	privacyCol := record.Column(10).(*array.String)
	inputCol := record.Column(11).(*array.String)
	outputCol := record.Column(12).(*array.String)

	results := make([]OperationRow, 0, record.NumRows())
	for i := 0; i < int(record.NumRows()); i++ {
		row := OperationRow{
			ID:           stringValue(idCol, i),
			LibraryType:  stringValue(libCol, i),
			Operation:    stringValue(opCol, i),
			Status:       stringValue(statusCol, i),
			Error:        stringValue(errCol, i),
			SessionID:    stringValue(sessionCol, i),
			UserIDHash:   stringValue(userCol, i),
			PrivacyLevel: stringValue(privacyCol, i),
		}
		if latencyCol != nil && !latencyCol.IsNull(i) {
			row.LatencyMs = latencyCol.Value(i)
		}
		if ts := stringValue(createdCol, i); ts != "" {
			if parsed, err := time.Parse(time.RFC3339Nano, ts); err == nil {
				row.CreatedAt = parsed
			}
		}
		if ts := stringValue(completedCol, i); ts != "" {
			if parsed, err := time.Parse(time.RFC3339Nano, ts); err == nil {
				row.CompletedAt = &parsed
			}
		}
		if payload := stringValue(inputCol, i); payload != "" {
			row.Input = parseJSON(payload)
		}
		if payload := stringValue(outputCol, i); payload != "" {
			row.Output = parseJSON(payload)
		}
		results = append(results, row)
	}
	return results
}

func stringValue(col *array.String, idx int) string {
	if col == nil || idx < 0 || idx >= col.Len() || col.IsNull(idx) {
		return ""
	}
	return col.Value(idx)
}

func parseJSON(raw string) map[string]any {
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil
	}
	return out
}
