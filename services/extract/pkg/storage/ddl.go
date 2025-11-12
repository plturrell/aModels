package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
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
	cmd := exec.CommandContext(ctx, "python3", "./scripts/utils/parse_hive_ddl.py", "--ddl", ddl)

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

func ddlToGraph(parsed ddlParseResult) ([]Node, []Edge) {
	var nodes []Node
	var edges []Edge
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
				nodes = append(nodes, Node{
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
			edges = append(edges, Edge{
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

		nodes = append(nodes, Node{
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
				"type":     normalizeColumnType(column.Type),
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
						edges = append(edges, Edge{
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
						edges = append(edges, Edge{
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

			nodes = append(nodes, Node{
				ID:    columnID,
				Type:  "column",
				Label: column.Name,
				Props: mapOrNil(columnProps),
			})
			edges = append(edges, Edge{
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
