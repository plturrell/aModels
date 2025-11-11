package connectors

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strings"
	"time"

	_ "github.com/lib/pq"           // PostgreSQL driver
	_ "github.com/go-sql-driver/mysql" // MySQL driver
	_ "github.com/mattn/go-sqlite3"    // SQLite driver

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// RelationalConnector connects to relational databases (PostgreSQL, MySQL, SQLite, etc.).
type RelationalConnector struct {
	config     map[string]interface{}
	logger     *log.Logger
	db         *sql.DB
	dbType     string // "postgres", "mysql", "sqlite"
	dsn        string
}

// NewRelationalConnector creates a new relational database connector.
func NewRelationalConnector(config map[string]interface{}, logger *log.Logger) *RelationalConnector {
	return &RelationalConnector{
		config: config,
		logger: logger,
	}
}

// Connect establishes connection to the relational database.
func (rc *RelationalConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	// Merge provided config
	for k, v := range config {
		rc.config[k] = v
	}

	// Determine database type
	dbType, _ := config["db_type"].(string)
	if dbType == "" {
		dbType, _ = config["database_type"].(string)
	}
	if dbType == "" {
		dbType = "postgres" // Default to PostgreSQL
	}
	rc.dbType = strings.ToLower(dbType)

	// Build DSN based on database type
	var dsn string
	switch rc.dbType {
	case "postgres", "postgresql":
		host := getString(config, "host", "localhost")
		port := getString(config, "port", "5432")
		user := getString(config, "user", "postgres")
		password := getString(config, "password", "")
		database := getString(config, "database", "postgres")
		sslmode := getString(config, "sslmode", "disable")
		
		// Try DATABASE_URL first
		if dbURL, ok := config["database_url"].(string); ok && dbURL != "" {
			dsn = dbURL
		} else {
			dsn = fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
				host, port, user, password, database, sslmode)
		}
	case "mysql":
		host := getString(config, "host", "localhost")
		port := getString(config, "port", "3306")
		user := getString(config, "user", "root")
		password := getString(config, "password", "")
		database := getString(config, "database", "mysql")
		
		if dbURL, ok := config["database_url"].(string); ok && dbURL != "" {
			dsn = dbURL
		} else {
			dsn = fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?parseTime=true",
				user, password, host, port, database)
		}
	case "sqlite", "sqlite3":
		path := getString(config, "path", getString(config, "database", ":memory:"))
		dsn = path
	default:
		return fmt.Errorf("unsupported database type: %s (supported: postgres, mysql, sqlite)", rc.dbType)
	}

	rc.dsn = dsn

	// Determine driver name
	var driverName string
	switch rc.dbType {
	case "postgres", "postgresql":
		driverName = "postgres"
	case "mysql":
		driverName = "mysql"
	case "sqlite", "sqlite3":
		driverName = "sqlite3"
	}

	// Open database connection
	var err error
	rc.db, err = sql.Open(driverName, dsn)
	if err != nil {
		return fmt.Errorf("failed to open database connection: %w", err)
	}

	// Set connection pool settings
	rc.db.SetMaxOpenConns(10)
	rc.db.SetMaxIdleConns(5)
	rc.db.SetConnMaxLifetime(5 * time.Minute)

	// Test connection
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := rc.db.PingContext(ctx); err != nil {
		rc.db.Close()
		return fmt.Errorf("failed to ping database: %w", err)
	}

	if rc.logger != nil {
		rc.logger.Printf("Connected to %s database", rc.dbType)
	}

	return nil
}

// DiscoverSchema discovers schema from the relational database.
func (rc *RelationalConnector) DiscoverSchema(ctx context.Context) (*agents.SourceSchema, error) {
	if rc.db == nil {
		return nil, fmt.Errorf("not connected to database")
	}

	// Get schema name (if specified)
	schemaName, _ := rc.config["schema"].(string)
	if schemaName == "" {
		// Default schema based on database type
		switch rc.dbType {
		case "postgres", "postgresql":
			schemaName = "public"
		case "mysql":
			schemaName = getString(rc.config, "database", "mysql")
		case "sqlite", "sqlite3":
			schemaName = "main"
		}
	}

	// Discover tables based on database type
	var tables []agents.TableDefinition
	var err error

	switch rc.dbType {
	case "postgres", "postgresql":
		tables, err = rc.discoverPostgresSchema(ctx, schemaName)
	case "mysql":
		tables, err = rc.discoverMySQLSchema(ctx, schemaName)
	case "sqlite", "sqlite3":
		tables, err = rc.discoverSQLiteSchema(ctx)
	default:
		return nil, fmt.Errorf("unsupported database type for schema discovery: %s", rc.dbType)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to discover schema: %w", err)
	}

	// Discover relations (foreign keys)
	relations, err := rc.discoverRelations(ctx, schemaName)
	if err != nil {
		if rc.logger != nil {
			rc.logger.Printf("Warning: Failed to discover relations: %v", err)
		}
		relations = []agents.RelationDefinition{}
	}

	schema := &agents.SourceSchema{
		SourceType: "relational",
		Tables:     tables,
		Relations:  relations,
		Metadata: map[string]interface{}{
			"database_type": rc.dbType,
			"schema":        schemaName,
			"dsn":           rc.maskDSN(rc.dsn),
		},
	}

	if rc.logger != nil {
		rc.logger.Printf("Discovered schema: %d tables, %d relations", len(tables), len(relations))
	}

	return schema, nil
}

// discoverPostgresSchema discovers schema from PostgreSQL.
func (rc *RelationalConnector) discoverPostgresSchema(ctx context.Context, schemaName string) ([]agents.TableDefinition, error) {
	query := `
		SELECT 
			t.table_name,
			c.column_name,
			c.data_type,
			c.is_nullable,
			c.column_default,
			COALESCE(pk.column_name, '') as primary_key
		FROM information_schema.tables t
		LEFT JOIN information_schema.columns c ON t.table_name = c.table_name 
			AND t.table_schema = c.table_schema
		LEFT JOIN (
			SELECT kcu.table_name, kcu.column_name, kcu.table_schema
			FROM information_schema.table_constraints tc
			JOIN information_schema.key_column_usage kcu 
				ON tc.constraint_name = kcu.constraint_name
				AND tc.table_schema = kcu.table_schema
			WHERE tc.constraint_type = 'PRIMARY KEY'
		) pk ON c.table_name = pk.table_name 
			AND c.column_name = pk.column_name
			AND c.table_schema = pk.table_schema
		WHERE t.table_schema = $1
			AND t.table_type = 'BASE TABLE'
		ORDER BY t.table_name, c.ordinal_position
	`

	rows, err := rc.db.QueryContext(ctx, query, schemaName)
	if err != nil {
		return nil, fmt.Errorf("failed to query schema: %w", err)
	}
	defer rows.Close()

	tableMap := make(map[string]*agents.TableDefinition)
	var currentTable *agents.TableDefinition

	for rows.Next() {
		var tableName, columnName, dataType, isNullable, columnDefault, primaryKey sql.NullString
		if err := rows.Scan(&tableName, &columnName, &dataType, &isNullable, &columnDefault, &primaryKey); err != nil {
			continue
		}

		if !tableName.Valid {
			continue
		}

		// Create or get table
		if currentTable == nil || currentTable.Name != tableName.String {
			currentTable = &agents.TableDefinition{
				Name:        tableName.String,
				Columns:     []agents.ColumnDefinition{},
				PrimaryKey:  []string{},
				ForeignKeys: []agents.ForeignKeyDefinition{},
				Metadata:    map[string]interface{}{},
			}
			tableMap[tableName.String] = currentTable
		}

		// Add column
		if columnName.Valid {
			nullable := isNullable.Valid && isNullable.String == "YES"
			col := agents.ColumnDefinition{
				Name:     columnName.String,
				Type:     rc.mapPostgresType(dataType.String),
				Nullable: nullable,
			}
			if columnDefault.Valid {
				col.Metadata = map[string]interface{}{
					"default": columnDefault.String,
				}
			}
			currentTable.Columns = append(currentTable.Columns, col)

			// Add to primary key if applicable
			if primaryKey.Valid && primaryKey.String == columnName.String {
				currentTable.PrimaryKey = append(currentTable.PrimaryKey, columnName.String)
			}
		}
	}

	// Convert map to slice
	tables := make([]agents.TableDefinition, 0, len(tableMap))
	for _, table := range tableMap {
		tables = append(tables, *table)
	}

	return tables, nil
}

// discoverMySQLSchema discovers schema from MySQL.
func (rc *RelationalConnector) discoverMySQLSchema(ctx context.Context, schemaName string) ([]agents.TableDefinition, error) {
	query := `
		SELECT 
			t.TABLE_NAME,
			c.COLUMN_NAME,
			c.DATA_TYPE,
			c.IS_NULLABLE,
			c.COLUMN_DEFAULT,
			CASE WHEN kcu.COLUMN_NAME IS NOT NULL THEN kcu.COLUMN_NAME ELSE '' END as PRIMARY_KEY
		FROM information_schema.TABLES t
		LEFT JOIN information_schema.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME 
			AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
		LEFT JOIN information_schema.KEY_COLUMN_USAGE kcu 
			ON c.TABLE_NAME = kcu.TABLE_NAME 
			AND c.COLUMN_NAME = kcu.COLUMN_NAME
			AND c.TABLE_SCHEMA = kcu.TABLE_SCHEMA
			AND kcu.CONSTRAINT_NAME = 'PRIMARY'
		WHERE t.TABLE_SCHEMA = ?
			AND t.TABLE_TYPE = 'BASE TABLE'
		ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
	`

	rows, err := rc.db.QueryContext(ctx, query, schemaName)
	if err != nil {
		return nil, fmt.Errorf("failed to query schema: %w", err)
	}
	defer rows.Close()

	tableMap := make(map[string]*agents.TableDefinition)
	var currentTable *agents.TableDefinition

	for rows.Next() {
		var tableName, columnName, dataType, isNullable, columnDefault, primaryKey sql.NullString
		if err := rows.Scan(&tableName, &columnName, &dataType, &isNullable, &columnDefault, &primaryKey); err != nil {
			continue
		}

		if !tableName.Valid {
			continue
		}

		// Create or get table
		if currentTable == nil || currentTable.Name != tableName.String {
			currentTable = &agents.TableDefinition{
				Name:        tableName.String,
				Columns:     []agents.ColumnDefinition{},
				PrimaryKey:  []string{},
				ForeignKeys: []agents.ForeignKeyDefinition{},
				Metadata:    map[string]interface{}{},
			}
			tableMap[tableName.String] = currentTable
		}

		// Add column
		if columnName.Valid {
			nullable := isNullable.Valid && isNullable.String == "YES"
			col := agents.ColumnDefinition{
				Name:     columnName.String,
				Type:     rc.mapMySQLType(dataType.String),
				Nullable: nullable,
			}
			if columnDefault.Valid {
				col.Metadata = map[string]interface{}{
					"default": columnDefault.String,
				}
			}
			currentTable.Columns = append(currentTable.Columns, col)

			// Add to primary key if applicable
			if primaryKey.Valid && primaryKey.String == columnName.String {
				currentTable.PrimaryKey = append(currentTable.PrimaryKey, columnName.String)
			}
		}
	}

	// Convert map to slice
	tables := make([]agents.TableDefinition, 0, len(tableMap))
	for _, table := range tableMap {
		tables = append(tables, *table)
	}

	return tables, nil
}

// discoverSQLiteSchema discovers schema from SQLite.
func (rc *RelationalConnector) discoverSQLiteSchema(ctx context.Context) ([]agents.TableDefinition, error) {
	query := `
		SELECT name FROM sqlite_master 
		WHERE type='table' AND name NOT LIKE 'sqlite_%'
		ORDER BY name
	`

	rows, err := rc.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query tables: %w", err)
	}
	defer rows.Close()

	var tableNames []string
	for rows.Next() {
		var tableName string
		if err := rows.Scan(&tableName); err != nil {
			continue
		}
		tableNames = append(tableNames, tableName)
	}

	// Get schema for each table
	var tables []agents.TableDefinition
	for _, tableName := range tableNames {
		// Use PRAGMA table_info to get column information
		pragmaQuery := fmt.Sprintf("PRAGMA table_info(%s)", tableName)
		pragmaRows, err := rc.db.QueryContext(ctx, pragmaQuery)
		if err != nil {
			continue
		}

		table := agents.TableDefinition{
			Name:        tableName,
			Columns:     []agents.ColumnDefinition{},
			PrimaryKey:  []string{},
			ForeignKeys: []agents.ForeignKeyDefinition{},
			Metadata:    map[string]interface{}{},
		}

		for pragmaRows.Next() {
			var cid int
			var name, dataType string
			var notNull, pk int
			var defaultValue sql.NullString

			if err := pragmaRows.Scan(&cid, &name, &dataType, &notNull, &defaultValue, &pk); err != nil {
				continue
			}

			col := agents.ColumnDefinition{
				Name:     name,
				Type:     rc.mapSQLiteType(dataType),
				Nullable: notNull == 0,
			}
			if defaultValue.Valid {
				col.Metadata = map[string]interface{}{
					"default": defaultValue.String,
				}
			}
			table.Columns = append(table.Columns, col)

			if pk > 0 {
				table.PrimaryKey = append(table.PrimaryKey, name)
			}
		}
		pragmaRows.Close()

		tables = append(tables, table)
	}

	return tables, nil
}

// discoverRelations discovers foreign key relationships.
func (rc *RelationalConnector) discoverRelations(ctx context.Context, schemaName string) ([]agents.RelationDefinition, error) {
	var query string
	var args []interface{}

	switch rc.dbType {
	case "postgres", "postgresql":
		query = `
			SELECT
				tc.table_name AS source_table,
				kcu.column_name AS source_column,
				ccu.table_name AS target_table,
				ccu.column_name AS target_column,
				tc.constraint_name
			FROM information_schema.table_constraints AS tc
			JOIN information_schema.key_column_usage AS kcu
				ON tc.constraint_name = kcu.constraint_name
				AND tc.table_schema = kcu.table_schema
			JOIN information_schema.constraint_column_usage AS ccu
				ON ccu.constraint_name = tc.constraint_name
				AND ccu.table_schema = tc.table_schema
			WHERE tc.constraint_type = 'FOREIGN KEY'
				AND tc.table_schema = $1
		`
		args = []interface{}{schemaName}
	case "mysql":
		query = `
			SELECT
				kcu.TABLE_NAME AS source_table,
				kcu.COLUMN_NAME AS source_column,
				kcu.REFERENCED_TABLE_NAME AS target_table,
				kcu.REFERENCED_COLUMN_NAME AS target_column,
				kcu.CONSTRAINT_NAME
			FROM information_schema.KEY_COLUMN_USAGE kcu
			WHERE kcu.TABLE_SCHEMA = ?
				AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
		`
		args = []interface{}{schemaName}
	case "sqlite", "sqlite3":
		// SQLite foreign keys are in sqlite_master
		// This is simplified - full implementation would parse CREATE TABLE statements
		return []agents.RelationDefinition{}, nil
	default:
		return []agents.RelationDefinition{}, nil
	}

	rows, err := rc.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var relations []agents.RelationDefinition
	for rows.Next() {
		var sourceTable, sourceColumn, targetTable, targetColumn, constraintName sql.NullString
		if err := rows.Scan(&sourceTable, &sourceColumn, &targetTable, &targetColumn, &constraintName); err != nil {
			continue
		}

		if sourceTable.Valid && targetTable.Valid {
			relations = append(relations, agents.RelationDefinition{
				FromTable:   sourceTable.String,
				ToTable:     targetTable.String,
				Type:        "foreign_key",
				FromColumns: []string{sourceColumn.String},
				ToColumns:   []string{targetColumn.String},
				Metadata: map[string]interface{}{
					"constraint_name": constraintName.String,
				},
			})
		}
	}

	return relations, nil
}

// ExtractData extracts data from the relational database.
// Query parameters:
//   - table: Table name to query
//   - schema: Schema name (optional, uses default if not provided)
//   - limit: Maximum number of rows to fetch (default: 1000)
//   - offset: Offset for pagination (default: 0)
//   - where: WHERE clause (optional, must be safe SQL)
//   - columns: Comma-separated list of columns to select (optional, defaults to *)
func (rc *RelationalConnector) ExtractData(ctx context.Context, query map[string]interface{}) ([]map[string]interface{}, error) {
	if rc.db == nil {
		return nil, fmt.Errorf("not connected to database")
	}

	tableName, ok := query["table"].(string)
	if !ok || tableName == "" {
		return nil, fmt.Errorf("table name is required")
	}

	schemaName, _ := query["schema"].(string)
	if schemaName == "" {
		switch rc.dbType {
		case "postgres", "postgresql":
			schemaName = "public"
		case "mysql":
			schemaName = getString(rc.config, "database", "mysql")
		case "sqlite", "sqlite3":
			schemaName = "main"
		}
	}

	limit := 1000
	if l, ok := query["limit"].(int); ok && l > 0 {
		limit = l
	}

	offset := 0
	if o, ok := query["offset"].(int); ok && o >= 0 {
		offset = o
	}

	columns := "*"
	if cols, ok := query["columns"].(string); ok && cols != "" {
		columns = cols
	}

	whereClause := ""
	if where, ok := query["where"].(string); ok && where != "" {
		// Basic safety check - in production, use parameterized queries
		whereClause = "WHERE " + where
	}

	// Build query based on database type
	var sqlQuery string
	switch rc.dbType {
	case "postgres", "postgresql":
		if schemaName != "" {
			tableName = fmt.Sprintf(`"%s"."%s"`, schemaName, tableName)
		}
		sqlQuery = fmt.Sprintf("SELECT %s FROM %s %s LIMIT %d OFFSET %d",
			columns, tableName, whereClause, limit, offset)
	case "mysql":
		if schemaName != "" {
			tableName = fmt.Sprintf("`%s`.`%s`", schemaName, tableName)
		}
		sqlQuery = fmt.Sprintf("SELECT %s FROM %s %s LIMIT %d OFFSET %d",
			columns, tableName, whereClause, limit, offset)
	case "sqlite", "sqlite3":
		sqlQuery = fmt.Sprintf("SELECT %s FROM %s %s LIMIT %d OFFSET %d",
			columns, tableName, whereClause, limit, offset)
	}

	rows, err := rc.db.QueryContext(ctx, sqlQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	columnsList, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	var results []map[string]interface{}
	for rows.Next() {
		values := make([]interface{}, len(columnsList))
		valuePtrs := make([]interface{}, len(columnsList))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			continue
		}

		row := make(map[string]interface{})
		for i, col := range columnsList {
			val := values[i]
			if val != nil {
				// Convert []byte to string for better JSON serialization
				if b, ok := val.([]byte); ok {
					row[col] = string(b)
				} else {
					row[col] = val
				}
			} else {
				row[col] = nil
			}
		}
		// Add metadata
		row["_table"] = tableName
		row["_schema"] = schemaName
		row["_database_type"] = rc.dbType

		results = append(results, row)
	}

	if rc.logger != nil {
		rc.logger.Printf("Extracted %d rows from table %s", len(results), tableName)
	}

	return results, nil
}

// Close closes the database connection.
func (rc *RelationalConnector) Close() error {
	if rc.db != nil {
		if rc.logger != nil {
			rc.logger.Printf("Closing database connection")
		}
		return rc.db.Close()
	}
	return nil
}

// Helper functions

func getString(m map[string]interface{}, key string, defaultValue string) string {
	if v, ok := m[key].(string); ok && v != "" {
		return v
	}
	return defaultValue
}

func (rc *RelationalConnector) mapPostgresType(dbType string) string {
	dbType = strings.ToLower(dbType)
	switch {
	case strings.Contains(dbType, "int"):
		return "integer"
	case strings.Contains(dbType, "decimal") || strings.Contains(dbType, "numeric") || strings.Contains(dbType, "real") || strings.Contains(dbType, "double"):
		return "decimal"
	case strings.Contains(dbType, "bool"):
		return "boolean"
	case strings.Contains(dbType, "date") && !strings.Contains(dbType, "time"):
		return "date"
	case strings.Contains(dbType, "time"):
		return "timestamp"
	case strings.Contains(dbType, "text") || strings.Contains(dbType, "varchar") || strings.Contains(dbType, "char"):
		return "string"
	case strings.Contains(dbType, "json"):
		return "json"
	case strings.Contains(dbType, "uuid"):
		return "uuid"
	default:
		return "string"
	}
}

func (rc *RelationalConnector) mapMySQLType(dbType string) string {
	dbType = strings.ToLower(dbType)
	switch {
	case strings.Contains(dbType, "int"):
		return "integer"
	case strings.Contains(dbType, "decimal") || strings.Contains(dbType, "numeric") || strings.Contains(dbType, "float") || strings.Contains(dbType, "double"):
		return "decimal"
	case strings.Contains(dbType, "bool") || strings.Contains(dbType, "bit"):
		return "boolean"
	case strings.Contains(dbType, "date") && !strings.Contains(dbType, "time"):
		return "date"
	case strings.Contains(dbType, "time"):
		return "timestamp"
	case strings.Contains(dbType, "text") || strings.Contains(dbType, "varchar") || strings.Contains(dbType, "char"):
		return "string"
	case strings.Contains(dbType, "json"):
		return "json"
	default:
		return "string"
	}
}

func (rc *RelationalConnector) mapSQLiteType(dbType string) string {
	dbType = strings.ToLower(dbType)
	switch {
	case strings.Contains(dbType, "int"):
		return "integer"
	case strings.Contains(dbType, "real") || strings.Contains(dbType, "float") || strings.Contains(dbType, "double"):
		return "decimal"
	case strings.Contains(dbType, "text") || strings.Contains(dbType, "varchar") || strings.Contains(dbType, "char"):
		return "string"
	case strings.Contains(dbType, "blob"):
		return "binary"
	case strings.Contains(dbType, "date") || strings.Contains(dbType, "time"):
		return "timestamp"
	default:
		return "string"
	}
}

func (rc *RelationalConnector) maskDSN(dsn string) string {
	// Mask password in DSN for logging
	if strings.Contains(dsn, "password=") {
		parts := strings.Split(dsn, " ")
		for i, part := range parts {
			if strings.HasPrefix(part, "password=") {
				parts[i] = "password=***"
			}
		}
		return strings.Join(parts, " ")
	}
	if strings.Contains(dsn, "@") {
		// For connection strings like user:pass@host
		parts := strings.Split(dsn, "@")
		if len(parts) > 0 {
			userPass := strings.Split(parts[0], ":")
			if len(userPass) > 1 {
				userPass[1] = "***"
				parts[0] = strings.Join(userPass, ":")
			}
		}
		return strings.Join(parts, "@")
	}
	return dsn
}

