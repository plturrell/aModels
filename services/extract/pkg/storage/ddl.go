package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

type ddlParseResult struct {
	Tables []ddlTable `json:"tables"`
}

type ddlTable struct {
	TableName       string           `json:"table_name"`
	Schema          string           `json:"schema"`
	Columns         []ddlColumn      `json:"columns"`
	TableProperties map[string]any   `json:"table_properties"`
	PartitionedBy   []map[string]any `json:"partitioned_by"`
	PrimaryKey      []string         `json:"primary_key"`
}

type ddlColumn struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	Size       any    `json:"size"`
	Nullable   bool   `json:"nullable"`
	Default    any    `json:"default"`
	Check      any    `json:"check"`
	Unique     bool   `json:"unique"`
	References any    `json:"references"`
}

func parseHiveDDL(ctx context.Context, ddl string) (ddlParseResult, error) {
	// Check if ddl is a file path (contains "/" or starts with "/")
	// If it's a file path, read the file content
	ddlContent := ddl
	if strings.Contains(ddl, "/") || strings.HasPrefix(ddl, "/") {
		// Check if file exists
		if _, err := os.Stat(ddl); err == nil {
			// Read file content
			content, err := os.ReadFile(ddl)
			if err != nil {
				return ddlParseResult{}, fmt.Errorf("read ddl file %q: %w", ddl, err)
			}
			ddlContent = string(content)
		}
	}

	// Determine Python script path - try multiple locations
	scriptPaths := []string{
		"/tmp/parse_hive_ddl.py", // Container temp location
		"./scripts/utils/parse_hive_ddl.py",
		"/workspace/services/extract/scripts/utils/parse_hive_ddl.py",
		filepath.Join(filepath.Dir(os.Args[0]), "../scripts/utils/parse_hive_ddl.py"),
	}

	var scriptPath string
	for _, path := range scriptPaths {
		if _, err := os.Stat(path); err == nil {
			scriptPath = path
			break
		}
	}

	if scriptPath == "" {
		// Fallback to relative path
		scriptPath = "./scripts/utils/parse_hive_ddl.py"
	}

	// Write DDL content to temp file to avoid "Argument list too long" error
	// Large DDL files exceed command-line argument limits
	tmpFile, err := os.CreateTemp("", "ddl_*.sql")
	if err != nil {
		return ddlParseResult{}, fmt.Errorf("create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name()) // Clean up temp file

	if _, err := tmpFile.WriteString(ddlContent); err != nil {
		tmpFile.Close()
		return ddlParseResult{}, fmt.Errorf("write temp file: %w", err)
	}
	tmpFile.Close()

	// Use --ddl-file to pass file path (avoids command-line length limits)
	cmd := exec.CommandContext(ctx, "python3", scriptPath, "--ddl-file", tmpFile.Name())

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return ddlParseResult{}, fmt.Errorf("parse hive ddl: %w, stderr: %s", err, strings.TrimSpace(string(exitErr.Stderr)))
		}
		return ddlParseResult{}, fmt.Errorf("parse hive ddl: %w", err)
	}

	var parsed ddlParseResult
	if err := json.Unmarshal(output, &parsed); err != nil {
		return ddlParseResult{}, fmt.Errorf("parse hive ddl: %w", err)
	}

	return parsed, nil
}

func ddlToGraph(parsed ddlParseResult) ([]graph.Node, []graph.Edge) {
	var nodes []graph.Node
	var edges []graph.Edge
	schemaNodes := make(map[string]bool) // Track created schema nodes

	for _, table := range parsed.Tables {
		if table.TableName == "" {
			continue
		}

		tableID := table.TableName
		schemaName := ""
		if schema := strings.TrimSpace(table.Schema); schema != "" {
			schemaName = schema
			tableID = fmt.Sprintf("%s.%s", schema, table.TableName)
		}

		// Create schema/database node if not already created
		if schemaName != "" {
			schemaID := fmt.Sprintf("schema:%s", schemaName)
			if !schemaNodes[schemaID] {
				nodes = append(nodes, graph.Node{
					ID:    schemaID,
					Type:  "database",
					Label: schemaName,
					Props: map[string]any{
						"schema": schemaName,
					},
				})
				schemaNodes[schemaID] = true
			}
			// Create CONTAINS relationship from schema to table
			edges = append(edges, graph.Edge{
				SourceID: schemaID,
				TargetID: tableID,
				Label:    "CONTAINS",
				Props: map[string]any{
					"source": "ddl",
				},
			})
		}

		tableProps := make(map[string]any)
		if schemaName != "" {
			tableProps["schema"] = schemaName
		}
		if len(table.TableProperties) > 0 {
			tableProps["properties"] = table.TableProperties
		}
		if len(table.PrimaryKey) > 0 {
			tableProps["primary_key"] = table.PrimaryKey
		}
		if len(table.PartitionedBy) > 0 {
			tableProps["partitioned_by"] = table.PartitionedBy
		}

		nodes = append(nodes, graph.Node{
			ID:    tableID,
			Type:  "table",
			Label: table.TableName,
			Props: mapOrNil(tableProps),
		})

		for _, column := range table.Columns {
			if column.Name == "" {
				continue
			}

			columnID := fmt.Sprintf("%s.%s", tableID, column.Name)
			columnProps := map[string]any{
				"type":     strings.ToLower(strings.TrimSpace(column.Type)),
				"nullable": column.Nullable,
			}
			if column.Default != nil {
				columnProps["default"] = column.Default
			}
			if column.Unique {
				columnProps["unique"] = true
			}
			if column.Size != nil {
				columnProps["size"] = column.Size
			}
			if column.Check != nil {
				columnProps["check"] = column.Check
			}
			if column.References != nil {
				columnProps["references"] = column.References
				// Create REFERENCES relationship if we can parse the reference
				if refMap, ok := column.References.(map[string]any); ok {
					if refTable, ok := refMap["table"].(string); ok {
						refTableID := refTable
						if refSchema, ok := refMap["schema"].(string); ok && refSchema != "" {
							refTableID = fmt.Sprintf("%s.%s", refSchema, refTable)
						}
						// Create REFERENCES edge from source table to referenced table
						edges = append(edges, graph.Edge{
							SourceID: tableID,
							TargetID: refTableID,
							Label:    "REFERENCES",
							Props: map[string]any{
								"foreign_key_column": column.Name,
								"referenced_column":  refMap["column"],
							},
						})
					}
				} else if refStr, ok := column.References.(string); ok && refStr != "" {
					// Try to parse string reference (format: "schema.table.column" or "table.column")
					parts := strings.Split(refStr, ".")
					if len(parts) >= 2 {
						refTableID := strings.Join(parts[:len(parts)-1], ".")
						edges = append(edges, graph.Edge{
							SourceID: tableID,
							TargetID: refTableID,
							Label:    "REFERENCES",
							Props: map[string]any{
								"foreign_key_column": column.Name,
								"referenced_column":  parts[len(parts)-1],
							},
						})
					}
				}
			}

			nodes = append(nodes, graph.Node{
				ID:    columnID,
				Type:  "column",
				Label: column.Name,
				Props: mapOrNil(columnProps),
			})
			edges = append(edges, graph.Edge{
				SourceID: tableID,
				TargetID: columnID,
				Label:    "HAS_COLUMN",
			})
		}
	}

	return nodes, edges
}

func mapOrNil(m map[string]any) map[string]any {
	if len(m) == 0 {
		return nil
	}
	return m
}

// MapOrNil returns nil if the map is empty, otherwise returns the map.
// This is useful for avoiding empty maps in JSON serialization.
func MapOrNil(m map[string]any) map[string]any {
	return mapOrNil(m)
}
