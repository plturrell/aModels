package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	flightGraphNodesPath = "graph/nodes"
	flightGraphEdgesPath = "graph/edges"
)

var (
	nodeSchema = arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "type", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "label", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "properties", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)

	edgeSchema = arrow.NewSchema([]arrow.Field{
		{Name: "source", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "target", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "label", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "properties", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
)

type extractFlightServer struct {
	addr    string
	core    flight.Server
	service *extractFlightService
	logger  *log.Logger
}

func newExtractFlightServer(logger *log.Logger) *extractFlightServer {
	service := &extractFlightService{
		BaseFlightServer: flight.BaseFlightServer{},
		allocator:        memory.NewGoAllocator(),
	}
	core := flight.NewServerWithMiddleware(nil)
	core.RegisterFlightService(service)

	return &extractFlightServer{
		core:    core,
		service: service,
		logger:  logger,
	}
}

func (s *extractFlightServer) Start(addr string) error {
	if s == nil {
		return nil
	}
	if err := s.core.Init(addr); err != nil {
		return fmt.Errorf("init flight server: %w", err)
	}
	s.addr = addr
	s.logger.Printf("extract Flight service listening on %s", addr)
	return s.core.Serve()
}

func (s *extractFlightServer) Shutdown() {
	if s == nil {
		return
	}
	s.core.Shutdown()
}

func (s *extractFlightServer) UpdateGraph(nodes []Node, edges []Edge) {
	if s == nil || s.service == nil {
		return
	}
	s.service.UpdateGraph(nodes, edges)
}

type extractFlightService struct {
	flight.BaseFlightServer
	allocator memory.Allocator

	mu    sync.RWMutex
	nodes []Node
	edges []Edge
}

func (s *extractFlightService) UpdateGraph(nodes []Node, edges []Edge) {
	s.mu.Lock()
	s.nodes = append([]Node(nil), nodes...)
	s.edges = append([]Edge(nil), edges...)
	s.mu.Unlock()
}

func (s *extractFlightService) ListFlights(_ *flight.Criteria, stream flight.FlightService_ListFlightsServer) error {
	infos, err := s.flightInfos()
	if err != nil {
		return err
	}
	for _, info := range infos {
		if err := stream.Send(info); err != nil {
			return err
		}
	}
	return nil
}

func (s *extractFlightService) GetFlightInfo(_ context.Context, descriptor *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	if descriptor == nil {
		return nil, status.Error(codes.InvalidArgument, "descriptor required")
	}
	if descriptor.Type != flight.DescriptorPATH || len(descriptor.Path) != 2 || descriptor.Path[0] != "graph" {
		return nil, status.Error(codes.NotFound, "unknown descriptor")
	}
	switch descriptor.Path[1] {
	case "nodes":
		infos, err := s.flightInfos()
		if err != nil {
			return nil, err
		}
		return infos[0], nil
	case "edges":
		infos, err := s.flightInfos()
		if err != nil {
			return nil, err
		}
		return infos[1], nil
	default:
		return nil, status.Error(codes.NotFound, "unknown descriptor")
	}
}

func (s *extractFlightService) DoGet(ticket *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	if ticket == nil {
		return status.Error(codes.InvalidArgument, "ticket required")
	}

	switch string(ticket.GetTicket()) {
	case flightGraphNodesPath:
		nodes := s.snapshotNodes()
		return s.writeNodes(stream, nodes)
	case flightGraphEdgesPath:
		edges := s.snapshotEdges()
		return s.writeEdges(stream, edges)
	default:
		return status.Error(codes.NotFound, "ticket not recognised")
	}
}

func (s *extractFlightService) snapshotNodes() []Node {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]Node(nil), s.nodes...)
}

func (s *extractFlightService) snapshotEdges() []Edge {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]Edge(nil), s.edges...)
}

func (s *extractFlightService) flightInfos() ([]*flight.FlightInfo, error) {
	s.mu.RLock()
	nodeCount := int64(len(s.nodes))
	edgeCount := int64(len(s.edges))
	s.mu.RUnlock()

	nodeDescriptor := &flight.FlightDescriptor{Type: flight.DescriptorPATH, Path: []string{"graph", "nodes"}}
	edgeDescriptor := &flight.FlightDescriptor{Type: flight.DescriptorPATH, Path: []string{"graph", "edges"}}

	nodeEndpoint := &flight.FlightEndpoint{Ticket: &flight.Ticket{Ticket: []byte(flightGraphNodesPath)}}
	edgeEndpoint := &flight.FlightEndpoint{Ticket: &flight.Ticket{Ticket: []byte(flightGraphEdgesPath)}}

	nodeInfo := &flight.FlightInfo{
		FlightDescriptor: nodeDescriptor,
		Endpoint:         []*flight.FlightEndpoint{nodeEndpoint},
		TotalRecords:     nodeCount,
		TotalBytes:       nodeCount,
	}

	edgeInfo := &flight.FlightInfo{
		FlightDescriptor: edgeDescriptor,
		Endpoint:         []*flight.FlightEndpoint{edgeEndpoint},
		TotalRecords:     edgeCount,
		TotalBytes:       edgeCount,
	}

	return []*flight.FlightInfo{nodeInfo, edgeInfo}, nil
}

func (s *extractFlightService) writeNodes(stream flight.FlightService_DoGetServer, nodes []Node) error {
	record, err := buildNodeRecord(s.allocator, nodes)
	if err != nil {
		return err
	}
	defer record.Release()

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(nodeSchema))
	defer writer.Close()

	if err := writer.Write(record); err != nil {
		return status.Errorf(codes.Internal, "write nodes record: %v", err)
	}
	return nil
}

func (s *extractFlightService) writeEdges(stream flight.FlightService_DoGetServer, edges []Edge) error {
	record, err := buildEdgeRecord(s.allocator, edges)
	if err != nil {
		return err
	}
	defer record.Release()

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(edgeSchema))
	defer writer.Close()

	if err := writer.Write(record); err != nil {
		return status.Errorf(codes.Internal, "write edges record: %v", err)
	}
	return nil
}

func buildNodeRecord(allocator memory.Allocator, nodes []Node) (arrow.Record, error) {
	builder := array.NewRecordBuilder(allocator, nodeSchema)
	defer builder.Release()

	idBuilder := builder.Field(0).(*array.StringBuilder)
	typeBuilder := builder.Field(1).(*array.StringBuilder)
	labelBuilder := builder.Field(2).(*array.StringBuilder)
	propsBuilder := builder.Field(3).(*array.StringBuilder)

	for _, node := range nodes {
		idBuilder.Append(node.ID)
		typeBuilder.Append(node.Type)
		labelBuilder.Append(node.Label)
		if len(node.Props) == 0 {
			propsBuilder.AppendNull()
			continue
		}
		payload, err := json.Marshal(node.Props)
		if err != nil {
			return nil, fmt.Errorf("marshal node properties: %w", err)
		}
		propsBuilder.Append(string(payload))
	}

	return builder.NewRecord(), nil
}

func buildEdgeRecord(allocator memory.Allocator, edges []Edge) (arrow.Record, error) {
	builder := array.NewRecordBuilder(allocator, edgeSchema)
	defer builder.Release()

	sourceBuilder := builder.Field(0).(*array.StringBuilder)
	targetBuilder := builder.Field(1).(*array.StringBuilder)
	labelBuilder := builder.Field(2).(*array.StringBuilder)
	propsBuilder := builder.Field(3).(*array.StringBuilder)

	for _, edge := range edges {
		sourceBuilder.Append(edge.SourceID)
		targetBuilder.Append(edge.TargetID)
		labelBuilder.Append(edge.Label)
		if len(edge.Props) == 0 {
			propsBuilder.AppendNull()
			continue
		}
		payload, err := json.Marshal(edge.Props)
		if err != nil {
			return nil, fmt.Errorf("marshal edge properties: %w", err)
		}
		propsBuilder.Append(string(payload))
	}

	return builder.NewRecord(), nil
}
