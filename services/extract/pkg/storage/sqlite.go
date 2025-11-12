package storage

import (
	"database/sql"
	"fmt"
	"strings"

	"github.com/plturrell/aModels/services/extract/pkg/utils"
	_ "github.com/mattn/go-sqlite3"
)

// SQLitePersistence is the persistence layer for SQLite.
type SQLitePersistence struct {
	db *sql.DB
}

// NewSQLitePersistence creates a new SQLite persistence layer.
func NewSQLitePersistence(path string) (*SQLitePersistence, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite database: %w", err)
	}

	return &SQLitePersistence{db: db}, nil
}

// SaveTable saves a table to SQLite.
func (p *SQLitePersistence) SaveTable(tableName string, data []map[string]any) error {
	if len(data) == 0 {
		return nil
	}

	// Sanitize table name to prevent SQL injection
	sanitizedTableName, err := utils.SanitizeIdentifier(tableName)
	if err != nil {
		return fmt.Errorf("invalid table name: %w", err)
	}

	// Create table
	var columns []string
	for key := range data[0] {
		// Sanitize column names
		sanitizedCol, err := utils.SanitizeIdentifier(key)
		if err != nil {
			return fmt.Errorf("invalid column name %s: %w", key, err)
		}
		columns = append(columns, sanitizedCol)
	}

	var createTableSQL strings.Builder
	createTableSQL.WriteString(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (", sanitizedTableName))
	for i, col := range columns {
		createTableSQL.WriteString(fmt.Sprintf("%s TEXT", col))
		if i < len(columns)-1 {
			createTableSQL.WriteString(", ")
		}
	}
	createTableSQL.WriteString(")")

	_, err := p.db.Exec(createTableSQL.String())
	if err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}

	// Insert data
	tx, err := p.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	if _, err := tx.Exec(fmt.Sprintf("DELETE FROM %s", sanitizedTableName)); err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to clear table %s: %w", sanitizedTableName, err)
	}

	var insertSQL strings.Builder
	insertSQL.WriteString(fmt.Sprintf("INSERT INTO %s (", sanitizedTableName))
	insertSQL.WriteString(strings.Join(columns, ", "))
	insertSQL.WriteString(") VALUES (")
	for i := range columns {
		insertSQL.WriteString("?")
		if i < len(columns)-1 {
			insertSQL.WriteString(", ")
		}
	}
	insertSQL.WriteString(")")

	stmt, err := tx.Prepare(insertSQL.String())
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, row := range data {
		var values []any
		for _, col := range columns {
			values = append(values, row[col])
		}
		_, err := stmt.Exec(values...)
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to insert row: %w", err)
		}
	}

	return tx.Commit()
}
