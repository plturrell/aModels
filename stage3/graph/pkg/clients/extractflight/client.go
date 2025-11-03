package extractflight

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"github.com/apache/arrow/go/v16/arrow/ipc"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	nodesPath = "graph/nodes"
	edgesPath = "graph/edges"
)

// GraphData holds node and edge rows fetched from the extract Flight server.
type GraphData struct {
	Nodes []GraphRow `json:"nodes"`
	Edges []GraphRow `json:"edges"`
}

// GraphRow captures a single row from the Flight dataset.
type GraphRow struct {
	ID         string         `json:"id"`
	Type       string         `json:"type,omitempty"`
	Label      string         `json:"label,omitempty"`
	Properties map[string]any `json:"properties,omitempty"`
	Raw        map[string]any `json:"-"`
}

// Fetch retrieves nodes and edges from the configured Flight endpoint.
func Fetch(ctx context.Context, addr string) (GraphData, error) {
	if addr == "" {
		return GraphData{}, fmt.Errorf("flight address is required")
	}

	client, err := flight.NewClientWithMiddlewareCtx(ctx, addr, nil, nil, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return GraphData{}, fmt.Errorf("dial flight endpoint: %w", err)
	}
	defer client.Close()

	nodes, err := fetchRecords(ctx, client, []string{"graph", "nodes"}, nodesPath)
	if err != nil {
		return GraphData{}, fmt.Errorf("fetch nodes: %w", err)
	}
	edges, err := fetchRecords(ctx, client, []string{"graph", "edges"}, edgesPath)
	if err != nil {
		return GraphData{}, fmt.Errorf("fetch edges: %w", err)
	}

	return GraphData{
		Nodes: nodes,
		Edges: edges,
	}, nil
}

func fetchRecords(ctx context.Context, client flight.Client, descriptorPath []string, ticket string) ([]GraphRow, error) {
	descriptor := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: descriptorPath,
	}

	info, err := client.GetFlightInfo(ctx, descriptor)
	if err != nil {
		return nil, err
	}
	if len(info.Endpoint) == 0 {
		return nil, nil
	}

	var rows []GraphRow
	alloc := memory.NewGoAllocator()

	for _, endpoint := range info.Endpoint {
		tkt := endpoint.Ticket
		if tkt == nil {
			tkt = &flight.Ticket{Ticket: []byte(ticket)}
		}
		stream, err := client.DoGet(ctx, tkt)
		if err != nil {
			return nil, err
		}
		reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(alloc))
		if err != nil {
			stream.CloseSend()
			return nil, err
		}
		for reader.Next() {
			record := reader.Record()
			rows = append(rows, recordToRows(record)...)
		}
		reader.Release()
		_ = stream.CloseSend()
	}
	return rows, nil
}

func recordToRows(record arrow.Record) []GraphRow {
	if record == nil {
		return nil
	}

	idCol := record.Column(0).(*array.String)
	typeCol := record.Column(1).(*array.String)
	labelCol := record.Column(2).(*array.String)
	propsCol := record.Column(3).(*array.String)

	rows := make([]GraphRow, 0, record.NumRows())
	for i := 0; i < int(record.NumRows()); i++ {
		row := GraphRow{
			ID:    safeString(idCol, i),
			Type:  safeString(typeCol, i),
			Label: safeString(labelCol, i),
		}
		if props := safeString(propsCol, i); props != "" {
			var parsed map[string]any
			if err := json.Unmarshal([]byte(props), &parsed); err == nil {
				row.Properties = parsed
			}
		}
		rows = append(rows, row)
	}
	return rows
}

func safeString(col *array.String, idx int) string {
	if col == nil || idx < 0 || idx >= col.Len() || col.IsNull(idx) {
		return ""
	}
	return col.Value(idx)
}
