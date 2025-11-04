package sqldatabase

import (
	"context"
	"database/sql"
	"fmt"
)

// Engine represents a database engine interface
type Engine interface {
	Dialect() string
	Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
	Close() error
}

// SQLDatabase wraps a database engine and provides helper methods
type SQLDatabase struct {
	engine Engine
	tables []string
}

// NewSQLDatabase creates a new SQLDatabase
func NewSQLDatabase(engine Engine, tables []string) (*SQLDatabase, error) {
	if engine == nil {
		return nil, fmt.Errorf("engine cannot be nil")
	}

	return &SQLDatabase{
		engine: engine,
		tables: tables,
	}, nil
}

// Dialect returns the SQL dialect
func (db *SQLDatabase) Dialect() string {
	return db.engine.Dialect()
}

// Query executes a SQL query
func (db *SQLDatabase) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	return db.engine.Query(ctx, query, args...)
}

// GetTableInfo returns table schema information
func (db *SQLDatabase) GetTableInfo(ctx context.Context, tables []string) (string, error) {
	return db.TableInfo(ctx, tables)
}

// TableInfo returns table schema information (alias for GetTableInfo)
func (db *SQLDatabase) TableInfo(ctx context.Context, tables []string) (string, error) {
	// Simplified implementation - returns a basic description
	if len(tables) == 0 {
		tables = db.tables
	}

	var info string
	for _, table := range tables {
		info += fmt.Sprintf("Table: %s\n", table)
	}

	return info, nil
}

// GetTableNames returns all table names
func (db *SQLDatabase) GetTableNames() []string {
	return db.tables
}

// Close closes the database connection
func (db *SQLDatabase) Close() error {
	if db.engine != nil {
		return db.engine.Close()
	}
	return nil
}

// QueryToString converts sql.Rows to a string representation
func QueryToString(rows *sql.Rows) (string, error) {
	if rows == nil {
		return "", nil
	}
	defer rows.Close()

	// Get column names
	columns, err := rows.Columns()
	if err != nil {
		return "", err
	}

	var result string
	result += fmt.Sprintf("Columns: %v\n", columns)

	// Read rows
	rowCount := 0
	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return "", err
		}

		result += fmt.Sprintf("Row %d: %v\n", rowCount, values)
		rowCount++
	}

	if err := rows.Err(); err != nil {
		return "", err
	}

	result += fmt.Sprintf("Total rows: %d\n", rowCount)
	return result, nil
}
