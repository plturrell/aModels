package testing

import (
	"context"
	"fmt"
	"math/rand"
)

// resolveForeignKeys ensures referenced tables have data.
func (sg *SampleGenerator) resolveForeignKeys(ctx context.Context, schema *TableSchema) error {
	for _, fk := range schema.ForeignKeys {
		// Check if referenced table exists in knowledge graph
		refSchema, exists := sg.knowledgeGraph.Tables[fk.ReferencedTable]
		if !exists {
			sg.logger.Printf("Warning: referenced table %s not found in knowledge graph", fk.ReferencedTable)
			continue
		}
		
		// Check if referenced table has data
		var count int
		query := fmt.Sprintf("SELECT COUNT(*) FROM %s", fk.ReferencedTable)
		if err := sg.db.QueryRowContext(ctx, query).Scan(&count); err != nil {
			// Table might not exist yet, that's okay
			sg.logger.Printf("Warning: could not check row count for %s: %v", fk.ReferencedTable, err)
			continue
		}
		
		// If no data exists, generate some
		if count == 0 {
			sg.logger.Printf("Generating reference data for foreign key table %s", fk.ReferencedTable)
			refConfig := &TableTestConfig{
				TableName: fk.ReferencedTable,
				RowCount:  50, // Default reference table size
			}
			
			refData, err := sg.GenerateSampleData(ctx, refConfig)
			if err != nil {
				return fmt.Errorf("generate reference data for %s: %w", fk.ReferencedTable, err)
			}
			
			if err := sg.insertData(ctx, fk.ReferencedTable, refData); err != nil {
				return fmt.Errorf("insert reference data for %s: %w", fk.ReferencedTable, err)
			}
		}
	}
	
	return nil
}

// resolveForeignKeyValue ensures a foreign key value exists in the referenced table.
func (sg *SampleGenerator) resolveForeignKeyValue(ctx context.Context, fk *ForeignKey, originalValue any) any {
	// Query for existing values in referenced table
	query := fmt.Sprintf("SELECT %s FROM %s LIMIT 1000", fk.ReferencedColumn, fk.ReferencedTable)
	rows, err := sg.db.QueryContext(ctx, query)
	if err != nil {
		// If query fails, return original value
		return originalValue
	}
	defer rows.Close()
	
	values := make([]any, 0)
	for rows.Next() {
		var val any
		if err := rows.Scan(&val); err == nil {
			values = append(values, val)
		}
	}
	
	// If we found values, randomly select one
	if len(values) > 0 {
		return values[rand.Intn(len(values))]
	}
	
	// Otherwise return original value
	return originalValue
}

