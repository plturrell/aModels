package main

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
	cmd := exec.CommandContext(ctx, "python3", "./scripts/parse_hive_ddl.py", "--ddl", ddl)

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

	for _, table := range parsed.Tables {
		if table.TableName == "" {
			continue
		}

		tableID := table.TableName
		if schema := strings.TrimSpace(table.Schema); schema != "" {
			tableID = fmt.Sprintf("%s.%s", schema, table.TableName)
		}

		tableProps := make(map[string]any)
		if schema := strings.TrimSpace(table.Schema); schema != "" {
			tableProps["schema"] = schema
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
				"type":     column.Type,
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
