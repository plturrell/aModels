package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// SAPBDCIntegration handles integration with SAP Business Data Cloud.
type SAPBDCIntegration struct {
	clientURL string
	logger    *log.Logger
	httpClient *http.Client
}

// NewSAPBDCIntegration creates a new SAP BDC integration.
func NewSAPBDCIntegration(logger *log.Logger) *SAPBDCIntegration {
	clientURL := os.Getenv("SAP_BDC_URL")
	if clientURL == "" {
		clientURL = "http://localhost:8083"
	}

	return &SAPBDCIntegration{
		clientURL: clientURL,
		logger:    logger,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ExtractFromSAPBDC extracts data from SAP Business Data Cloud and converts to graph format.
func (s *SAPBDCIntegration) ExtractFromSAPBDC(
	ctx context.Context,
	formationID string,
	sourceSystem string,
	dataProductID string,
	spaceID string,
	database string,
	projectID string,
	systemID string,
) ([]Node, []Edge, error) {
	s.logger.Printf("Extracting from SAP BDC: formation=%s, source=%s", formationID, sourceSystem)

	// Call SAP BDC service
	reqBody := map[string]any{
		"formation_id":    formationID,
		"source_system":    sourceSystem,
		"include_views":    true,
	}

	if dataProductID != "" {
		reqBody["data_product_id"] = dataProductID
	}
	if spaceID != "" {
		reqBody["space_id"] = spaceID
	}
	if database != "" {
		reqBody["database"] = database
	}

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.clientURL+"/extract", 
		strings.NewReader(string(reqJSON)))
	if err != nil {
		return nil, nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("SAP BDC API error: %d - %s", resp.StatusCode, string(body))
	}

	var extractResp struct {
		Success      bool                    `json:"success"`
		Schema       *SAPBDCSchema           `json:"schema,omitempty"`
		Error        string                  `json:"error,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&extractResp); err != nil {
		return nil, nil, fmt.Errorf("decode response: %w", err)
	}

	if !extractResp.Success {
		return nil, nil, fmt.Errorf("SAP BDC extraction failed: %s", extractResp.Error)
	}

	if extractResp.Schema == nil {
		return nil, nil, fmt.Errorf("no schema returned from SAP BDC")
	}

	// Convert SAP BDC schema to graph format
	nodes, edges := s.convertSAPBDCSchemaToGraph(extractResp.Schema, projectID, systemID)

	return nodes, edges, nil
}

// SAPBDCSchema represents schema from SAP BDC (matches service.go format).
type SAPBDCSchema struct {
	Database string                   `json:"database"`
	Schema   string                   `json:"schema"`
	Tables   []SAPBDCTableInfo        `json:"tables"`
	Views    []SAPBDCViewInfo         `json:"views"`
	Metadata map[string]any            `json:"metadata"`
}

// SAPBDCTableInfo represents table information from SAP BDC.
type SAPBDCTableInfo struct {
	Name        string                   `json:"name"`
	Schema      string                   `json:"schema"`
	Columns     []SAPBDCColumnInfo       `json:"columns"`
	PrimaryKeys []string                 `json:"primary_keys"`
	ForeignKeys []SAPBDCForeignKeyInfo   `json:"foreign_keys"`
	Metadata    map[string]any           `json:"metadata"`
}

// SAPBDCColumnInfo represents column information from SAP BDC.
type SAPBDCColumnInfo struct {
	Name     string         `json:"name"`
	DataType string         `json:"data_type"`
	Nullable bool           `json:"nullable"`
	Default  any            `json:"default,omitempty"`
	Comment  string         `json:"comment,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// SAPBDCViewInfo represents view information from SAP BDC.
type SAPBDCViewInfo struct {
	Name       string                   `json:"name"`
	Schema     string                   `json:"schema"`
	Definition string                   `json:"definition"`
	Columns    []SAPBDCColumnInfo       `json:"columns"`
	Metadata   map[string]any           `json:"metadata"`
}

// SAPBDCForeignKeyInfo represents foreign key information from SAP BDC.
type SAPBDCForeignKeyInfo struct {
	Name            string `json:"name"`
	Column          string `json:"column"`
	ReferencedTable string `json:"referenced_table"`
	ReferencedColumn string `json:"referenced_column"`
}

// convertSAPBDCSchemaToGraph converts SAP BDC schema to aModels graph format.
func (s *SAPBDCIntegration) convertSAPBDCSchemaToGraph(
	schema *SAPBDCSchema,
	projectID string,
	systemID string,
) ([]Node, []Edge) {
	nodes := []Node{}
	edges := []Edge{}

	// Create database/schema node
	dbNode := Node{
		ID:    fmt.Sprintf("sap_bdc:%s:%s", schema.Database, schema.Schema),
		Type:  "database",
		Label: fmt.Sprintf("%s.%s", schema.Database, schema.Schema),
		Props: map[string]any{
			"database":   schema.Database,
			"schema":     schema.Schema,
			"source":     "sap_bdc",
			"project_id": projectID,
			"system_id":  systemID,
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
					"database":    schema.Database,
					"schema":      schema.Schema,
					"table_name":  table.Name,
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
					"source":            "sap_bdc",
					"foreign_key":       fk.Name,
					"column":            fk.Column,
					"referenced_column": fk.ReferencedColumn,
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
					"database":    schema.Database,
					"schema":      schema.Schema,
					"view_name":   view.Name,
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

	return nodes, edges
}

