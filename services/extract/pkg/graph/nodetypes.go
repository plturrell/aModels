package graph

// Node type constants for the extract service knowledge graph.
// These constants provide a centralized definition of all valid node types,
// reducing errors from typos and improving maintainability.

// Database schema node types
const (
	// NodeTypeTable represents a database table
	NodeTypeTable = "table"
	// NodeTypeColumn represents a database column
	NodeTypeColumn = "column"
	// NodeTypeView represents a database view
	NodeTypeView = "view"
	// NodeTypeDatabase represents a database or schema container
	NodeTypeDatabase = "database"
	// NodeTypeSchema represents a database schema
	NodeTypeSchema = "schema"
)

// Control-M workflow node types
const (
	// NodeTypeControlMJob represents a Control-M job definition
	NodeTypeControlMJob = "control-m-job"
	// NodeTypeControlMCalendar represents a Control-M calendar
	NodeTypeControlMCalendar = "control-m-calendar"
	// NodeTypeControlMCondition represents a Control-M condition (event)
	NodeTypeControlMCondition = "control-m-condition"
)

// Petri net and workflow node types
const (
	// NodeTypeSQL represents a SQL query or SQL-related node
	NodeTypeSQL = "sql"
	// NodeTypePetriPlace represents a Petri net place (state/condition)
	NodeTypePetriPlace = "petri_place"
	// NodeTypePetriTransition represents a Petri net transition (job execution)
	NodeTypePetriTransition = "petri_transition"
	// NodeTypePetriSubprocess represents a Petri net subprocess
	NodeTypePetriSubprocess = "petri_subprocess"
	// NodeTypePetriNet represents a Petri net container
	NodeTypePetriNet = "petri_net"
	// NodeTypeCondition represents a workflow condition
	NodeTypeCondition = "condition"
	// NodeTypeJob represents a workflow job
	NodeTypeJob = "job"
	// NodeTypeInitial represents an initial place in a Petri net
	NodeTypeInitial = "initial"
	// NodeTypeInput represents an input place
	NodeTypeInput = "input"
	// NodeTypeOutput represents an output place
	NodeTypeOutput = "output"
)

// Document and content node types
const (
	// NodeTypeDocument represents a document node
	NodeTypeDocument = "document"
	// NodeTypeText represents a text content node
	NodeTypeText = "text"
)

// System and organizational node types
const (
	// NodeTypeProject represents a project container
	NodeTypeProject = "project"
	// NodeTypeSystem represents a system container
	NodeTypeSystem = "system"
	// NodeTypeInformationSystem represents an information system
	NodeTypeInformationSystem = "information-system"
	// NodeTypeRoot represents a root node
	NodeTypeRoot = "root"
)

// Agent and process node types
const (
	// NodeTypeAgent represents an agent node
	NodeTypeAgent = "agent"
	// NodeTypeSignavioProcess represents a Signavio process
	NodeTypeSignavioProcess = "signavio-process"
	// NodeTypeSignavioLane represents a Signavio lane
	NodeTypeSignavioLane = "signavio-lane"
	// NodeTypeSpawn represents a spawn node
	NodeTypeSpawn = "spawn"
	// NodeTypeUnknown represents an unknown or unclassified node type
	NodeTypeUnknown = "unknown"
)

// Node type categories for organization and validation
const (
	CategoryDatabase = "database"
	CategoryWorkflow = "workflow"
	CategoryDocument = "document"
	CategorySystem   = "system"
	CategoryAgent    = "agent"
	CategoryUnknown  = "unknown"
)

// allNodeTypes contains all valid node types for validation
var allNodeTypes = map[string]bool{
	NodeTypeTable:                true,
	NodeTypeColumn:               true,
	NodeTypeView:                 true,
	NodeTypeDatabase:             true,
	NodeTypeSchema:               true,
	NodeTypeControlMJob:          true,
	NodeTypeControlMCalendar:     true,
	NodeTypeControlMCondition:    true,
	NodeTypeSQL:                  true,
	NodeTypePetriPlace:           true,
	NodeTypePetriTransition:      true,
	NodeTypePetriSubprocess:      true,
	NodeTypePetriNet:             true,
	NodeTypeCondition:            true,
	NodeTypeJob:                  true,
	NodeTypeInitial:              true,
	NodeTypeInput:                true,
	NodeTypeOutput:               true,
	NodeTypeDocument:             true,
	NodeTypeText:                 true,
	NodeTypeProject:              true,
	NodeTypeSystem:               true,
	NodeTypeInformationSystem:    true,
	NodeTypeRoot:                 true,
	NodeTypeAgent:                true,
	NodeTypeSignavioProcess:      true,
	NodeTypeSignavioLane:          true,
	NodeTypeSpawn:                true,
	NodeTypeUnknown:              true,
}

// nodeTypeCategories maps node types to their categories
var nodeTypeCategories = map[string]string{
	NodeTypeTable:                CategoryDatabase,
	NodeTypeColumn:               CategoryDatabase,
	NodeTypeView:                 CategoryDatabase,
	NodeTypeDatabase:             CategoryDatabase,
	NodeTypeSchema:               CategoryDatabase,
	NodeTypeControlMJob:          CategoryWorkflow,
	NodeTypeControlMCalendar:     CategoryWorkflow,
	NodeTypeControlMCondition:    CategoryWorkflow,
	NodeTypeSQL:                  CategoryWorkflow,
	NodeTypePetriPlace:           CategoryWorkflow,
	NodeTypePetriTransition:      CategoryWorkflow,
	NodeTypePetriSubprocess:      CategoryWorkflow,
	NodeTypePetriNet:             CategoryWorkflow,
	NodeTypeCondition:            CategoryWorkflow,
	NodeTypeJob:                  CategoryWorkflow,
	NodeTypeInitial:              CategoryWorkflow,
	NodeTypeInput:                CategoryWorkflow,
	NodeTypeOutput:               CategoryWorkflow,
	NodeTypeDocument:             CategoryDocument,
	NodeTypeText:                 CategoryDocument,
	NodeTypeProject:              CategorySystem,
	NodeTypeSystem:               CategorySystem,
	NodeTypeInformationSystem:    CategorySystem,
	NodeTypeRoot:                 CategorySystem,
	NodeTypeAgent:                CategoryAgent,
	NodeTypeSignavioProcess:      CategoryAgent,
	NodeTypeSignavioLane:         CategoryAgent,
	NodeTypeSpawn:                CategoryAgent,
	NodeTypeUnknown:              CategoryUnknown,
}

// IsValidNodeType checks if a node type string is valid
func IsValidNodeType(nodeType string) bool {
	return allNodeTypes[nodeType]
}

// GetAllNodeTypes returns a slice of all valid node types
func GetAllNodeTypes() []string {
	types := make([]string, 0, len(allNodeTypes))
	for nodeType := range allNodeTypes {
		types = append(types, nodeType)
	}
	return types
}

// GetNodeTypeCategory returns the category for a given node type
func GetNodeTypeCategory(nodeType string) string {
	if category, ok := nodeTypeCategories[nodeType]; ok {
		return category
	}
	return CategoryUnknown
}

// GetNodeTypesByCategory returns all node types in a given category
func GetNodeTypesByCategory(category string) []string {
	var types []string
	for nodeType, cat := range nodeTypeCategories {
		if cat == category {
			types = append(types, nodeType)
		}
	}
	return types
}

