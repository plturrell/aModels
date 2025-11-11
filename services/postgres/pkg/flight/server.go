package flight

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"github.com/plturrell/aModels/services/postgres/pkg/models"
	"github.com/plturrell/aModels/services/postgres/pkg/repository"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	operationsPath = "operations/logs"
)

var operationsSchema = arrow.NewSchema([]arrow.Field{
	{Name: "id", Type: arrow.BinaryTypes.String, Nullable: false},
	{Name: "library_type", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "operation", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "status", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "error", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "latency_ms", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
	{Name: "created_at", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "completed_at", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "session_id", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "user_id_hash", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "privacy_level", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "input", Type: arrow.BinaryTypes.String, Nullable: true},
	{Name: "output", Type: arrow.BinaryTypes.String, Nullable: true},
}, nil)

// Server wraps an Arrow Flight server exposing lang operations datasets.
type Server struct {
	addr    string
	core    flight.Server
	service *operationsService
}

// New constructs a Flight server when addr is provided. Passing an empty address returns nil.
func New(addr string, repo *repository.OperationsRepository, maxRows int) (*Server, error) {
	if strings.TrimSpace(addr) == "" {
		return nil, nil
	}
	if repo == nil {
		return nil, fmt.Errorf("flight server requires repository")
	}
	if maxRows <= 0 {
		maxRows = 200
	}

	service := &operationsService{
		repo:      repo,
		maxRows:   maxRows,
		allocator: memory.NewGoAllocator(),
	}

	core := flight.NewServerWithMiddleware(nil)
	core.RegisterFlightService(service)

	if err := core.Init(addr); err != nil {
		return nil, fmt.Errorf("initialise flight server: %w", err)
	}

	return &Server{
		addr:    addr,
		core:    core,
		service: service,
	}, nil
}

// Addr reports the bound address.
func (s *Server) Addr() string {
	if s == nil {
		return ""
	}
	return s.addr
}

// Serve blocks handling Flight requests.
func (s *Server) Serve() error {
	if s == nil || s.core == nil {
		return nil
	}
	return s.core.Serve()
}

// Shutdown gracefully stops the Flight server.
func (s *Server) Shutdown() {
	if s == nil || s.core == nil {
		return
	}
	s.core.Shutdown()
}

type operationsService struct {
	flight.BaseFlightServer
	allocator memory.Allocator
	repo      *repository.OperationsRepository
	maxRows   int
}

func (s *operationsService) ListFlights(_ *flight.Criteria, stream flight.FlightService_ListFlightsServer) error {
	infos := s.flightInfos()
	for _, info := range infos {
		if err := stream.Send(info); err != nil {
			return err
		}
	}
	return nil
}

func (s *operationsService) GetFlightInfo(_ context.Context, descriptor *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	if descriptor == nil {
		return nil, status.Error(codes.InvalidArgument, "descriptor required")
	}
	if descriptor.Type != flight.DescriptorPATH {
		return nil, status.Error(codes.InvalidArgument, "descriptor must use PATH type")
	}
	if len(descriptor.Path) != 2 || descriptor.Path[0] != "operations" || descriptor.Path[1] != "logs" {
		return nil, status.Error(codes.NotFound, "unknown descriptor")
	}
	infos := s.flightInfos()
	return infos[0], nil
}

func (s *operationsService) DoGet(ticket *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	if ticket == nil {
		return status.Error(codes.InvalidArgument, "ticket required")
	}
	if string(ticket.GetTicket()) != operationsPath {
		return status.Error(codes.NotFound, "ticket not recognised")
	}

	ctx := stream.Context()
	ops, _, err := s.repo.ListOperations(ctx, repository.ListFilters{PageSize: s.maxRows})
	if err != nil {
		return status.Errorf(codes.Internal, "list operations: %v", err)
	}

	record, err := buildOperationsRecord(s.allocator, ops)
	if err != nil {
		return status.Errorf(codes.Internal, "build record: %v", err)
	}
	defer record.Release()

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(operationsSchema))
	defer writer.Close()

	if err := writer.Write(record); err != nil {
		return status.Errorf(codes.Internal, "write record: %v", err)
	}
	return nil
}

func (s *operationsService) flightInfos() []*flight.FlightInfo {
	descriptor := &flight.FlightDescriptor{Type: flight.DescriptorPATH, Path: []string{"operations", "logs"}}
	endpoint := &flight.FlightEndpoint{Ticket: &flight.Ticket{Ticket: []byte(operationsPath)}}

	return []*flight.FlightInfo{
		{
			FlightDescriptor: descriptor,
			Endpoint:         []*flight.FlightEndpoint{endpoint},
			TotalRecords:     -1,
			TotalBytes:       -1,
		},
	}
}

func buildOperationsRecord(alloc memory.Allocator, ops []*models.LangOperation) (arrow.Record, error) {
	builder := array.NewRecordBuilder(alloc, operationsSchema)
	defer builder.Release()

	idBuilder := builder.Field(0).(*array.StringBuilder)
	libraryBuilder := builder.Field(1).(*array.StringBuilder)
	operationBuilder := builder.Field(2).(*array.StringBuilder)
	statusBuilder := builder.Field(3).(*array.StringBuilder)
	errorBuilder := builder.Field(4).(*array.StringBuilder)
	latencyBuilder := builder.Field(5).(*array.Int64Builder)
	createdBuilder := builder.Field(6).(*array.StringBuilder)
	completedBuilder := builder.Field(7).(*array.StringBuilder)
	sessionBuilder := builder.Field(8).(*array.StringBuilder)
	userBuilder := builder.Field(9).(*array.StringBuilder)
	privacyBuilder := builder.Field(10).(*array.StringBuilder)
	inputBuilder := builder.Field(11).(*array.StringBuilder)
	outputBuilder := builder.Field(12).(*array.StringBuilder)

	for _, op := range ops {
		idBuilder.Append(op.ID)
		libraryBuilder.Append(op.LibraryType)
		operationBuilder.Append(op.Operation)
		statusBuilder.Append(statusToString(op.Status))
		errorBuilder.Append(op.Error)
		latencyBuilder.Append(op.LatencyMs)
		createdBuilder.Append(op.CreatedAt.Format(time.RFC3339Nano))
		if op.CompletedAt != nil {
			completedBuilder.Append(op.CompletedAt.Format(time.RFC3339Nano))
		} else {
			completedBuilder.AppendNull()
		}
		sessionBuilder.Append(op.SessionID)
		userBuilder.Append(op.UserIDHash)
		privacyBuilder.Append(op.PrivacyLevel)

		if payload, err := json.Marshal(op.Input); err == nil {
			inputBuilder.Append(string(payload))
		} else {
			return nil, fmt.Errorf("marshal input: %w", err)
		}
		if payload, err := json.Marshal(op.Output); err == nil {
			outputBuilder.Append(string(payload))
		} else {
			return nil, fmt.Errorf("marshal output: %w", err)
		}
	}

	return builder.NewRecord(), nil
}

func statusToString(status models.OperationStatus) string {
	switch status {
	case models.OperationStatusRunning:
		return "running"
	case models.OperationStatusSuccess:
		return "success"
	case models.OperationStatusError:
		return "error"
	default:
		return "unspecified"
	}
}
