package sapbdc

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// Service represents the SAP BDC inbound source service.
type Service struct {
	client *Client
	logger *log.Logger
}

// NewService creates a new SAP BDC service.
func NewService(client *Client, logger *log.Logger) *Service {
	return &Service{
		client: client,
		logger: logger,
	}
}

// ExtractRequest represents a request to extract data from SAP BDC.
type ExtractRequest struct {
	FormationID     string            `json:"formation_id"`
	SourceSystem   string            `json:"source_system"`
	DataProductID  string            `json:"data_product_id,omitempty"`
	SpaceID        string            `json:"space_id,omitempty"`
	Database       string            `json:"database,omitempty"`
	IncludeViews   bool              `json:"include_views,omitempty"`
	Options        map[string]any    `json:"options,omitempty"`
}

// ExtractResponse represents the response from extraction.
type ExtractResponse struct {
	Success       bool              `json:"success"`
	Formation     *Formation        `json:"formation,omitempty"`
	DataProducts  []DataProduct     `json:"data_products,omitempty"`
	Schema        *Schema           `json:"schema,omitempty"`
	Metadata      map[string]any    `json:"metadata,omitempty"`
	Error         string            `json:"error,omitempty"`
}

// Extract extracts data and schema from SAP Business Data Cloud.
func (s *Service) Extract(ctx context.Context, req ExtractRequest) (*ExtractResponse, error) {
	s.logger.Printf("Starting SAP BDC extraction for formation: %s, source: %s", req.FormationID, req.SourceSystem)

	// Get formation details
	formation, err := s.client.GetFormation(ctx)
	if err != nil {
		return &ExtractResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to get formation: %v", err),
		}, err
	}

	response := &ExtractResponse{
		Success:   true,
		Formation: formation,
		Metadata:  make(map[string]any),
	}

	// If specific data product requested, get its details
	if req.DataProductID != "" {
		product, err := s.client.GetDataProduct(ctx, req.DataProductID)
		if err != nil {
			response.Success = false
			response.Error = fmt.Sprintf("Failed to get data product: %v", err)
			return response, err
		}
		response.DataProducts = []DataProduct{*product}
	} else {
		// List all data products
		products, err := s.client.ListDataProducts(ctx)
		if err != nil {
			response.Success = false
			response.Error = fmt.Sprintf("Failed to list data products: %v", err)
			return response, err
		}
		response.DataProducts = products
	}

	// Extract schema from either Datasphere or HANA Data Lake
	if req.SpaceID != "" || req.Database != "" {
		schema, err := s.client.ExtractSchema(ctx, req.SpaceID, req.Database)
		if err != nil {
			response.Success = false
			response.Error = fmt.Sprintf("Failed to extract schema: %v", err)
			return response, err
		}
		response.Schema = schema
	}

	// Add metadata
	response.Metadata["extraction_time"] = time.Now().UTC().Format(time.RFC3339)
	response.Metadata["source_system"] = req.SourceSystem
	response.Metadata["formation_id"] = req.FormationID

	return response, nil
}

// ConvertToGraphFormat converts SAP BDC schema to aModels graph format.
func (s *Service) ConvertToGraphFormat(schema *Schema, projectID, systemID string) ([]Node, []Edge, error) {
	nodes := []Node{}
	edges := []Edge{}

	// Create database/schema node
	dbNode := Node{
		ID:    fmt.Sprintf("sap_bdc:%s:%s", schema.Database, schema.Schema),
		Type:  "database",
		Label: fmt.Sprintf("%s.%s", schema.Database, schema.Schema),
		Props: map[string]any{
			"database": schema.Database,
			"schema":   schema.Schema,
			"source":   "sap_bdc",
		},
	}
	nodes = append(nodes, dbNode)

	// Process tables
	for _, table := range schema.Tables {
		tableNode := Node{
			ID:    fmt.Sprintf("sap_bdc:table:%s:%s:%s", schema.Database, schema.Schema, table.Name),
			Type:  "table",
			Label: table.Name,
			Props: map[string]any{
				"database":   schema.Database,
				"schema":     schema.Schema,
				"table_name": table.Name,
				"source":     "sap_bdc",
				"project_id": projectID,
				"system_id":  systemID,
			},
		}

		// Add table metadata
		if table.Metadata != nil {
			for k, v := range table.Metadata {
				tableNode.Props[k] = v
			}
		}

		nodes = append(nodes, tableNode)

		// Create edge from database to table
		edges = append(edges, Edge{
			SourceID: dbNode.ID,
			TargetID: tableNode.ID,
			Label:    "CONTAINS",
			Props:    map[string]any{"source": "sap_bdc"},
		})

		// Process columns
		for _, column := range table.Columns {
			columnNode := Node{
				ID:    fmt.Sprintf("sap_bdc:column:%s:%s:%s:%s", schema.Database, schema.Schema, table.Name, column.Name),
				Type:  "column",
				Label: column.Name,
				Props: map[string]any{
					"database":   schema.Database,
					"schema":     schema.Schema,
					"table_name": table.Name,
					"column_name": column.Name,
					"data_type":   column.DataType,
					"nullable":    column.Nullable,
					"source":      "sap_bdc",
					"project_id":  projectID,
					"system_id":   systemID,
				},
			}

			if column.Default != nil {
				columnNode.Props["default"] = column.Default
			}
			if column.Comment != "" {
				columnNode.Props["comment"] = column.Comment
			}
			if column.Metadata != nil {
				for k, v := range column.Metadata {
					columnNode.Props[k] = v
				}
			}

			nodes = append(nodes, columnNode)

			// Create edge from table to column
			edges = append(edges, Edge{
				SourceID: tableNode.ID,
				TargetID: columnNode.ID,
				Label:    "HAS_COLUMN",
				Props:    map[string]any{"source": "sap_bdc"},
			})
		}

		// Process foreign keys
		for _, fk := range table.ForeignKeys {
			referencedTableID := fmt.Sprintf("sap_bdc:table:%s:%s:%s", schema.Database, schema.Schema, fk.ReferencedTable)
			edges = append(edges, Edge{
				SourceID: tableNode.ID,
				TargetID: referencedTableID,
				Label:    "REFERENCES",
				Props: map[string]any{
					"source":             "sap_bdc",
					"foreign_key":        fk.Name,
					"column":             fk.Column,
					"referenced_column":  fk.ReferencedColumn,
				},
			})
		}
	}

	// Process views
	for _, view := range schema.Views {
		viewNode := Node{
			ID:    fmt.Sprintf("sap_bdc:view:%s:%s:%s", schema.Database, schema.Schema, view.Name),
			Type:  "view",
			Label: view.Name,
			Props: map[string]any{
				"database":   schema.Database,
				"schema":     schema.Schema,
				"view_name":  view.Name,
				"definition": view.Definition,
				"source":     "sap_bdc",
				"project_id": projectID,
				"system_id":  systemID,
			},
		}

		if view.Metadata != nil {
			for k, v := range view.Metadata {
				viewNode.Props[k] = v
			}
		}

		nodes = append(nodes, viewNode)

		// Create edge from database to view
		edges = append(edges, Edge{
			SourceID: dbNode.ID,
			TargetID: viewNode.ID,
			Label:    "CONTAINS",
			Props:    map[string]any{"source": "sap_bdc"},
		})

		// Process view columns
		for _, column := range view.Columns {
			columnNode := Node{
				ID:    fmt.Sprintf("sap_bdc:column:%s:%s:%s:%s", schema.Database, schema.Schema, view.Name, column.Name),
				Type:  "column",
				Label: column.Name,
				Props: map[string]any{
					"database":   schema.Database,
					"schema":     schema.Schema,
					"view_name":  view.Name,
					"column_name": column.Name,
					"data_type":   column.DataType,
					"nullable":    column.Nullable,
					"source":      "sap_bdc",
					"project_id":  projectID,
					"system_id":   systemID,
				},
			}

			nodes = append(nodes, columnNode)

			// Create edge from view to column
			edges = append(edges, Edge{
				SourceID: viewNode.ID,
				TargetID: columnNode.ID,
				Label:    "HAS_COLUMN",
				Props:    map[string]any{"source": "sap_bdc"},
			})
		}
	}

	return nodes, edges, nil
}

// Node represents a graph node (shared with extract service).
type Node struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Label string         `json:"label"`
	Props map[string]any `json:"props,omitempty"`
}

// Edge represents a graph edge (shared with extract service).
type Edge struct {
	SourceID string         `json:"source_id"`
	TargetID string         `json:"target_id"`
	Label    string         `json:"label"`
	Props    map[string]any `json:"props,omitempty"`
}

