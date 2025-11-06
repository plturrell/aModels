package agents

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"
)

// DataIngestionAgent autonomously ingests data from source systems.
type DataIngestionAgent struct {
	ID            string
	SourceType    string // "murex", "sap_gl", "bcrs", "rco", "axiom"
	Connector     SourceConnector
	Mapper        SchemaMapper
	GraphClient   GraphClient
	logger        *log.Logger
	lastRun       time.Time
	stats         IngestionStats
}

// SourceConnector defines the interface for source system connectors.
type SourceConnector interface {
	Connect(ctx context.Context, config map[string]interface{}) error
	DiscoverSchema(ctx context.Context) (*SourceSchema, error)
	ExtractData(ctx context.Context, query map[string]interface{}) ([]map[string]interface{}, error)
	Close() error
}

// SchemaMapper maps source schemas to knowledge graph schemas.
type SchemaMapper interface {
	MapSchema(ctx context.Context, sourceSchema *SourceSchema) (*GraphSchema, error)
	MapData(ctx context.Context, sourceData []map[string]interface{}, mapping *MappingRules) ([]GraphNode, []GraphEdge, error)
}

// GraphClient defines the interface for knowledge graph operations.
type GraphClient interface {
	UpsertNodes(ctx context.Context, nodes []GraphNode) error
	UpsertEdges(ctx context.Context, edges []GraphEdge) error
	Query(ctx context.Context, cypher string, params map[string]interface{}) ([]map[string]interface{}, error)
}

// SourceSchema represents the schema of a source system.
type SourceSchema struct {
	SourceType string
	Tables     []TableDefinition
	Relations  []RelationDefinition
	Metadata   map[string]interface{}
}

// TableDefinition represents a table in a source system.
type TableDefinition struct {
	Name        string
	Columns     []ColumnDefinition
	PrimaryKey  []string
	ForeignKeys []ForeignKeyDefinition
	Metadata    map[string]interface{}
}

// ColumnDefinition represents a column in a source table.
type ColumnDefinition struct {
	Name        string
	Type        string
	Nullable    bool
	Default     interface{}
	Constraints map[string]interface{}
	Metadata    map[string]interface{}
}

// RelationDefinition represents a relationship between tables.
type RelationDefinition struct {
	FromTable   string
	ToTable     string
	Type        string
	FromColumns []string
	ToColumns   []string
	Metadata    map[string]interface{}
}

// ForeignKeyDefinition represents a foreign key relationship.
type ForeignKeyDefinition struct {
	Name            string
	ReferencedTable string
	Columns         []string
	ReferencedColumns []string
}

// GraphSchema represents the knowledge graph schema.
type GraphSchema struct {
	NodeLabels []string
	EdgeTypes  []string
	Properties map[string]PropertyDefinition
}

// PropertyDefinition represents a property in the graph.
type PropertyDefinition struct {
	Name     string
	Type     string
	Required bool
	Indexed  bool
}

// MappingRules defines how source data maps to graph nodes and edges.
type MappingRules struct {
	NodeMappings  []NodeMapping
	EdgeMappings  []EdgeMapping
	Transformations []Transformation
	Version       string
	Confidence    float64
}

// NodeMapping maps a source table to graph nodes.
type NodeMapping struct {
	SourceTable    string
	TargetLabel    string
	ColumnMappings []ColumnMapping
	Filters        map[string]interface{}
}

// ColumnMapping maps a source column to a graph property.
type ColumnMapping struct {
	SourceColumn string
	TargetProperty string
	Transformation string
	Default      interface{}
}

// EdgeMapping maps source relationships to graph edges.
type EdgeMapping struct {
	SourceRelation RelationDefinition
	TargetType     string
	PropertyMappings []PropertyMapping
}

// PropertyMapping maps source properties to edge properties.
type PropertyMapping struct {
	Source string
	Target string
}

// Transformation defines a data transformation.
type Transformation struct {
	Type        string // "cast", "format", "aggregate", "custom"
	Source      string
	Target      string
	Function    string
	Parameters  map[string]interface{}
}

// GraphNode represents a node in the knowledge graph.
type GraphNode struct {
	ID         string
	Labels     []string
	Properties map[string]interface{}
}

// GraphEdge represents an edge in the knowledge graph.
type GraphEdge struct {
	ID         string
	Type       string
	StartNode  string
	EndNode    string
	Properties map[string]interface{}
}

// IngestionStats tracks ingestion statistics.
type IngestionStats struct {
	TotalRuns        int
	SuccessfulRuns   int
	FailedRuns       int
	RecordsIngested  int64
	NodesCreated     int64
	EdgesCreated     int64
	LastSuccess      time.Time
	LastError        string
	AverageDuration  time.Duration
}

// NewDataIngestionAgent creates a new data ingestion agent.
func NewDataIngestionAgent(
	id string,
	sourceType string,
	connector SourceConnector,
	mapper SchemaMapper,
	graphClient GraphClient,
	logger *log.Logger,
) *DataIngestionAgent {
	return &DataIngestionAgent{
		ID:          id,
		SourceType:  sourceType,
		Connector:   connector,
		Mapper:      mapper,
		GraphClient: graphClient,
		logger:      logger,
		stats:       IngestionStats{},
	}
}

// Ingest performs autonomous data ingestion.
func (agent *DataIngestionAgent) Ingest(ctx context.Context, config map[string]interface{}) error {
	startTime := time.Now()
	agent.stats.TotalRuns++

	if agent.logger != nil {
		agent.logger.Printf("Starting autonomous ingestion from %s", agent.SourceType)
	}

	// Step 1: Connect to source system
	if err := agent.Connector.Connect(ctx, config); err != nil {
		agent.recordError("connection failed", err)
		return fmt.Errorf("failed to connect to source: %w", err)
	}
	defer agent.Connector.Close()

	// Step 2: Discover schema
	sourceSchema, err := agent.Connector.DiscoverSchema(ctx)
	if err != nil {
		agent.recordError("schema discovery failed", err)
		return fmt.Errorf("failed to discover schema: %w", err)
	}

	if agent.logger != nil {
		agent.logger.Printf("Discovered schema: %d tables, %d relations", len(sourceSchema.Tables), len(sourceSchema.Relations))
	}

	// Step 3: Map schema to graph schema
	graphSchema, err := agent.Mapper.MapSchema(ctx, sourceSchema)
	if err != nil {
		agent.recordError("schema mapping failed", err)
		return fmt.Errorf("failed to map schema: %w", err)
	}

	// Step 4: Get or create mapping rules
	mappingRules, err := agent.getOrCreateMappingRules(ctx, sourceSchema, graphSchema)
	if err != nil {
		agent.recordError("mapping rules failed", err)
		return fmt.Errorf("failed to get mapping rules: %w", err)
	}

	// Step 5: Extract data from source
	var allNodes []GraphNode
	var allEdges []GraphEdge
	totalRecords := int64(0)

	for _, table := range sourceSchema.Tables {
		// Extract data for this table
		query := map[string]interface{}{
			"table": table.Name,
			"limit": 1000, // Batch size
		}

		sourceData, err := agent.Connector.ExtractData(ctx, query)
		if err != nil {
			agent.logger.Printf("Warning: Failed to extract data from %s: %v", table.Name, err)
			continue
		}

		totalRecords += int64(len(sourceData))

		// Map data to graph nodes and edges
		nodes, edges, err := agent.Mapper.MapData(ctx, sourceData, mappingRules)
		if err != nil {
			agent.logger.Printf("Warning: Failed to map data from %s: %v", table.Name, err)
			continue
		}

		allNodes = append(allNodes, nodes...)
		allEdges = append(allEdges, edges...)
	}

	// Step 6: Upsert to knowledge graph
	if len(allNodes) > 0 {
		if err := agent.GraphClient.UpsertNodes(ctx, allNodes); err != nil {
			agent.recordError("node upsert failed", err)
			return fmt.Errorf("failed to upsert nodes: %w", err)
		}
	}

	if len(allEdges) > 0 {
		if err := agent.GraphClient.UpsertEdges(ctx, allEdges); err != nil {
			agent.recordError("edge upsert failed", err)
			return fmt.Errorf("failed to upsert edges: %w", err)
		}
	}

	// Step 7: Update statistics
	duration := time.Since(startTime)
	agent.stats.SuccessfulRuns++
	agent.stats.RecordsIngested += totalRecords
	agent.stats.NodesCreated += int64(len(allNodes))
	agent.stats.EdgesCreated += int64(len(allEdges))
	agent.stats.LastSuccess = time.Now()
	agent.stats.AverageDuration = (agent.stats.AverageDuration*time.Duration(agent.stats.SuccessfulRuns-1) + duration) / time.Duration(agent.stats.SuccessfulRuns)
	agent.lastRun = time.Now()

	if agent.logger != nil {
		agent.logger.Printf("Ingestion completed: %d records, %d nodes, %d edges in %v",
			totalRecords, len(allNodes), len(allEdges), duration)
	}

	return nil
}

// getOrCreateMappingRules retrieves existing mapping rules or creates new ones.
func (agent *DataIngestionAgent) getOrCreateMappingRules(ctx context.Context, sourceSchema *SourceSchema, graphSchema *GraphSchema) (*MappingRules, error) {
	// In production, would query knowledge graph for existing rules
	// For now, create default mapping rules
	return agent.createDefaultMappingRules(sourceSchema, graphSchema), nil
}

// createDefaultMappingRules creates default mapping rules from schema.
func (agent *DataIngestionAgent) createDefaultMappingRules(sourceSchema *SourceSchema, graphSchema *GraphSchema) *MappingRules {
	rules := &MappingRules{
		NodeMappings:   []NodeMapping{},
		EdgeMappings:  []EdgeMapping{},
		Transformations: []Transformation{},
		Version:       "1.0.0",
		Confidence:    0.7, // Initial confidence
	}

	// Create node mappings for each table
	for _, table := range sourceSchema.Tables {
		nodeMapping := NodeMapping{
			SourceTable:    table.Name,
			TargetLabel:    agent.inferLabel(table.Name),
			ColumnMappings: []ColumnMapping{},
		}

		// Map columns
		for _, column := range table.Columns {
			nodeMapping.ColumnMappings = append(nodeMapping.ColumnMappings, ColumnMapping{
				SourceColumn:   column.Name,
				TargetProperty: agent.inferProperty(column.Name),
				Transformation: "",
			})
		}

		rules.NodeMappings = append(rules.NodeMappings, nodeMapping)
	}

	// Create edge mappings for relationships
	for _, relation := range sourceSchema.Relations {
		edgeMapping := EdgeMapping{
			SourceRelation: relation,
			TargetType:     agent.inferEdgeType(relation.Type),
			PropertyMappings: []PropertyMapping{},
		}

		rules.EdgeMappings = append(rules.EdgeMappings, edgeMapping)
	}

	return rules
}

// inferLabel infers a graph label from a table name.
func (agent *DataIngestionAgent) inferLabel(tableName string) string {
	// Simple inference - in production would use ML/NLP
	// Convert table name to PascalCase label
	return strings.Title(strings.ReplaceAll(tableName, "_", ""))
}

// inferProperty infers a property name from a column name.
func (agent *DataIngestionAgent) inferProperty(columnName string) string {
	// Convert to camelCase
	return strings.ToLower(columnName[0:1]) + columnName[1:]
}

// inferEdgeType infers an edge type from a relation type.
func (agent *DataIngestionAgent) inferEdgeType(relationType string) string {
	return strings.ToUpper(relationType)
}

// GetStats returns ingestion statistics.
func (agent *DataIngestionAgent) GetStats() IngestionStats {
	return agent.stats
}

// recordError records an error in statistics.
func (agent *DataIngestionAgent) recordError(context string, err error) {
	agent.stats.FailedRuns++
	agent.stats.LastError = fmt.Sprintf("%s: %v", context, err)
	if agent.logger != nil {
		agent.logger.Printf("Ingestion error (%s): %v", context, err)
	}
}

