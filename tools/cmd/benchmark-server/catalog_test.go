package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai_benchmarks/internal/catalog/flightcatalog"
	"ai_benchmarks/internal/localai"
	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightdefs"
)

func TestHandleAgentCatalogIncludesEnrichment(t *testing.T) {
	stub := newTestFlightStub(t)
	defer stub.Release()

	server := flight.NewServerWithMiddleware(nil)
	server.RegisterFlightService(stub)
	if err := server.Init("localhost:0"); err != nil {
		t.Fatalf("init flight server: %v", err)
	}
	defer server.Shutdown()
	go func() { _ = server.Serve() }()

	handler := handleAgentCatalog(server.Addr().String())
	req := httptest.NewRequest(http.MethodGet, "/api/v1/agent-catalog", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	if _, ok := payload["Suites"]; !ok {
		t.Fatalf("expected Suites key in payload: %v", payload)
	}
	if _, ok := payload["Tools"]; !ok {
		t.Fatalf("expected Tools key in payload: %v", payload)
	}
	if summary, ok := payload["agent_catalog_summary"].(string); !ok || summary == "" {
		t.Fatalf("expected agent_catalog_summary string, got %T (%v)", payload["agent_catalog_summary"], payload["agent_catalog_summary"])
	}
	stats, ok := payload["agent_catalog_stats"].(map[string]any)
	if !ok {
		t.Fatalf("expected agent_catalog_stats object, got %T", payload["agent_catalog_stats"])
	}
	if suiteCount, ok := stats["suite_count"].(float64); !ok || suiteCount != 1 {
		t.Fatalf("expected suite_count 1, got %v", stats["suite_count"])
	}
	if _, ok := payload["agent_catalog_context"].(string); !ok {
		t.Fatalf("expected agent_catalog_context string, got %T", payload["agent_catalog_context"])
	}
}

func TestEnhancedInferenceCatalogEnrichment(t *testing.T) {
	stub := newTestFlightStub(t)
	defer stub.Release()

	server := flight.NewServerWithMiddleware(nil)
	server.RegisterFlightService(stub)
	if err := server.Init("localhost:0"); err != nil {
		t.Fatalf("init flight server: %v", err)
	}
	defer server.Shutdown()
	go func() { _ = server.Serve() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	cat, err := flightcatalog.Fetch(ctx, server.Addr().String())
	cancel()
	if err != nil {
		t.Fatalf("fetch catalog: %v", err)
	}

	engine := localai.NewEnhancedInferenceEngine(localai.NewClient("http://localhost", ""), localai.WithAgentCatalog(&cat))
	view := engine.CatalogEnrichment()
	if view.Summary == "" {
		t.Fatalf("expected summary to be populated")
	}
	if len(view.UniqueTools) == 0 {
		t.Fatalf("expected unique tools list to be populated")
	}
	if view.Stats.SuiteCount != 1 {
		t.Fatalf("expected suite count 1, got %d", view.Stats.SuiteCount)
	}
	if view.Prompt == "" {
		t.Fatalf("expected prompt context to be populated")
	}
}

type testFlightStub struct {
	flight.BaseFlightServer
	toolsRecord  arrow.Record
	suitesRecord arrow.Record
}

func newTestFlightStub(t *testing.T) *testFlightStub {
	t.Helper()
	alloc := memory.NewGoAllocator()

	toolsSchema := arrow.NewSchema([]arrow.Field{
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "description", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
	toolsBuilder := array.NewRecordBuilder(alloc, toolsSchema)
	toolsBuilder.Field(0).(*array.StringBuilder).Append("search_documents")
	toolsBuilder.Field(1).(*array.StringBuilder).Append("Search gateway document retrieval")
	toolsRecord := toolsBuilder.NewRecord()
	toolsBuilder.Release()

	suitesSchema := arrow.NewSchema([]arrow.Field{
		{Name: "suite", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "tool_count", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "tool_names", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "implementation", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "version", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "attached_at", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
	suitesBuilder := array.NewRecordBuilder(alloc, suitesSchema)
	suitesBuilder.Field(0).(*array.StringBuilder).Append("agentic-suite")
	suitesBuilder.Field(1).(*array.Int64Builder).Append(2)
	suitesBuilder.Field(2).(*array.StringBuilder).Append("search_documents,maths_execute_operation")
	suitesBuilder.Field(3).(*array.StringBuilder).Append("agentic-mcp")
	suitesBuilder.Field(4).(*array.StringBuilder).Append("0.1.0")
	suitesBuilder.Field(5).(*array.StringBuilder).Append(time.Now().UTC().Format(time.RFC3339Nano))
	suitesRecord := suitesBuilder.NewRecord()
	suitesBuilder.Release()

	return &testFlightStub{
		toolsRecord:  toolsRecord,
		suitesRecord: suitesRecord,
	}
}

func (s *testFlightStub) Release() {
	if s.toolsRecord != nil {
		s.toolsRecord.Release()
	}
	if s.suitesRecord != nil {
		s.suitesRecord.Release()
	}
}

func (s *testFlightStub) ListFlights(_ *flight.Criteria, stream flight.FlightService_ListFlightsServer) error {
	if err := stream.Send(s.buildFlightInfo(flightdefs.AgentToolsPath, s.toolsRecord.NumRows())); err != nil {
		return err
	}
	return stream.Send(s.buildFlightInfo(flightdefs.ServiceSuitesPath, s.suitesRecord.NumRows()))
}

func (s *testFlightStub) GetFlightInfo(_ context.Context, descriptor *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	path := strings.Join(descriptor.Path, "/")
	switch path {
	case flightdefs.AgentToolsPath:
		return s.buildFlightInfo(path, s.toolsRecord.NumRows()), nil
	case flightdefs.ServiceSuitesPath:
		return s.buildFlightInfo(path, s.suitesRecord.NumRows()), nil
	default:
		return nil, fmt.Errorf("unknown descriptor %s", path)
	}
}

func (s *testFlightStub) DoGet(ticket *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	switch string(ticket.GetTicket()) {
	case flightdefs.AgentToolsPath:
		return s.writeRecord(stream, s.toolsRecord)
	case flightdefs.ServiceSuitesPath:
		return s.writeRecord(stream, s.suitesRecord)
	default:
		return fmt.Errorf("unknown ticket")
	}
}

func (s *testFlightStub) writeRecord(stream flight.FlightService_DoGetServer, record arrow.Record) error {
	writer := flight.NewRecordWriter(stream, ipc.WithSchema(record.Schema()))
	defer writer.Close()
	return writer.Write(record)
}

func (s *testFlightStub) buildFlightInfo(path string, rows int64) *flight.FlightInfo {
	return &flight.FlightInfo{
		FlightDescriptor: &flight.FlightDescriptor{Type: flight.DescriptorPATH, Path: strings.Split(path, "/")},
		Endpoint: []*flight.FlightEndpoint{{
			Location: []*flight.Location{},
			Ticket:   &flight.Ticket{Ticket: []byte(path)},
		}},
		TotalRecords: rows,
		TotalBytes:   rows,
	}
}
