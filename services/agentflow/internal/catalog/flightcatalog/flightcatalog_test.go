package flightcatalog_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/internal/catalog/flightcatalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightdefs"
)

func TestFetchCatalog(t *testing.T) {
	ctx := context.Background()

	stub, err := newStubFlightService()
	if err != nil {
		t.Fatalf("new stub flight service: %v", err)
	}
	defer stub.Close()

	server := flight.NewServerWithMiddleware(nil)
	server.RegisterFlightService(stub)
	if err := server.Init("localhost:0"); err != nil {
		t.Fatalf("init flight server: %v", err)
	}
	defer server.Shutdown()
	go func() { _ = server.Serve() }()

	catalog, err := flightcatalog.Fetch(ctx, server.Addr().String())
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if len(catalog.Suites) != 1 {
		t.Fatalf("expected 1 suite, got %d", len(catalog.Suites))
	}
	if catalog.Suites[0].Name != "agentic" {
		t.Errorf("unexpected suite name %q", catalog.Suites[0].Name)
	}
	if len(catalog.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(catalog.Tools))
	}
	if catalog.Tools[0].Name != "search_documents" {
		t.Errorf("unexpected tool name %q", catalog.Tools[0].Name)
	}
}

func TestFetchCatalogMissingAddr(t *testing.T) {
	if _, err := flightcatalog.Fetch(context.Background(), ""); err == nil {
		t.Fatalf("expected error for empty address")
	}
}

type stubFlightService struct {
	flight.BaseFlightServer
	toolsRecord  arrow.Record
	suitesRecord arrow.Record
}

func newStubFlightService() (*stubFlightService, error) {
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
	suitesBuilder.Field(0).(*array.StringBuilder).Append("agentic")
	suitesBuilder.Field(1).(*array.Int64Builder).Append(1)
	suitesBuilder.Field(2).(*array.StringBuilder).Append("search_documents")
	suitesBuilder.Field(3).(*array.StringBuilder).Append("agentic-mcp")
	suitesBuilder.Field(4).(*array.StringBuilder).Append("0.1.0")
	suitesBuilder.Field(5).(*array.StringBuilder).Append(time.Date(2025, 10, 30, 11, 0, 0, 0, time.UTC).Format(time.RFC3339Nano))
	suitesRecord := suitesBuilder.NewRecord()
	suitesBuilder.Release()

	return &stubFlightService{
		toolsRecord:  toolsRecord,
		suitesRecord: suitesRecord,
	}, nil
}

func (s *stubFlightService) Close() {
	if s.toolsRecord != nil {
		s.toolsRecord.Release()
	}
	if s.suitesRecord != nil {
		s.suitesRecord.Release()
	}
}

func (s *stubFlightService) ListFlights(_ *flight.Criteria, stream flight.FlightService_ListFlightsServer) error {
	if err := stream.Send(s.buildFlightInfo(flightdefs.AgentToolsPath, s.toolsRecord.NumRows())); err != nil {
		return err
	}
	return stream.Send(s.buildFlightInfo(flightdefs.ServiceSuitesPath, s.suitesRecord.NumRows()))
}

func (s *stubFlightService) GetFlightInfo(_ context.Context, descriptor *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	if descriptor == nil {
		return nil, fmt.Errorf("descriptor required")
	}
	path := strings.Join(descriptor.Path, "/")
	switch path {
	case flightdefs.AgentToolsPath:
		return s.buildFlightInfo(path, s.toolsRecord.NumRows()), nil
	case flightdefs.ServiceSuitesPath:
		return s.buildFlightInfo(path, s.suitesRecord.NumRows()), nil
	default:
		return nil, fmt.Errorf("unknown descriptor")
	}
}

func (s *stubFlightService) DoGet(ticket *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	switch string(ticket.GetTicket()) {
	case flightdefs.AgentToolsPath:
		return s.writeRecord(stream, s.toolsRecord)
	case flightdefs.ServiceSuitesPath:
		return s.writeRecord(stream, s.suitesRecord)
	default:
		return fmt.Errorf("unknown ticket")
	}
}

func (s *stubFlightService) writeRecord(stream flight.FlightService_DoGetServer, record arrow.Record) error {
	writer := flight.NewRecordWriter(stream, ipc.WithSchema(record.Schema()))
	defer writer.Close()
	return writer.Write(record)
}

func (s *stubFlightService) buildFlightInfo(path string, rows int64) *flight.FlightInfo {
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
