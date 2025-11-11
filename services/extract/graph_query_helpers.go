package main

import (
	"fmt"
	"log"
)

// GraphQueryHelpers provides helper functions for common knowledge graph queries.
type GraphQueryHelpers struct {
	logger *log.Logger
}

// NewGraphQueryHelpers creates a new graph query helpers instance.
func NewGraphQueryHelpers(logger *log.Logger) *GraphQueryHelpers {
	return &GraphQueryHelpers{logger: logger}
}

// QueryPetriNets returns a Cypher query to find all Petri nets.
func (gqh *GraphQueryHelpers) QueryPetriNets() string {
	return `MATCH (n)
WHERE n.type = 'petri_net'
RETURN n.id as id, n.label as label, n.properties_json as properties
ORDER BY n.label`
}

// QueryPetriNetTransitions returns a Cypher query to find transitions with SQL subprocesses.
func (gqh *GraphQueryHelpers) QueryPetriNetTransitions(petriNetID string) string {
	return fmt.Sprintf(`MATCH (pn:Node)-[:HAS_TRANSITION]->(t:Node)-[:HAS_SUBPROCESS]->(s:Node)
WHERE pn.type = 'petri_net' AND pn.id = '%s'
  AND t.type = 'petri_transition'
  AND s.type = 'petri_subprocess'
  AND s.properties_json.subprocess_type = 'sql'
RETURN t.label as transition, s.properties_json.content as sql_query
ORDER BY t.label`, petriNetID)
}

// QueryWorkflowPaths returns a Cypher query to find workflow paths in a Petri net.
func (gqh *GraphQueryHelpers) QueryWorkflowPaths(petriNetID string) string {
	return fmt.Sprintf(`MATCH path = (p1:Node)-[:PETRI_ARC]->(t:Node)-[:PETRI_ARC]->(p2:Node)
WHERE p1.type = 'petri_place' 
  AND t.type = 'petri_transition'
  AND p2.type = 'petri_place'
  AND EXISTS((pn:Node)-[:HAS_PLACE]->(p1))
  AND EXISTS((pn:Node)-[:HAS_PLACE]->(p2))
  AND EXISTS((pn:Node)-[:HAS_TRANSITION]->(t))
  AND pn.id = '%s'
RETURN p1.label as source_place, t.label as transition, p2.label as target_place
ORDER BY p1.label, t.label`, petriNetID)
}

// QueryTransactionTables returns a Cypher query to find transaction tables.
func (gqh *GraphQueryHelpers) QueryTransactionTables() string {
	return `MATCH (n:Node)
WHERE n.type = 'table'
  AND n.properties_json.table_classification = 'transaction'
RETURN n.label as table_name, 
       n.properties_json.classification_confidence as confidence,
       n.properties_json.classification_evidence as evidence
ORDER BY confidence DESC`
}

// QueryProcessingSequences returns a Cypher query to find table processing sequences.
func (gqh *GraphQueryHelpers) QueryProcessingSequences() string {
	return `MATCH (a:Node)-[r:RELATIONSHIP]->(b:Node)
WHERE r.label = 'PROCESSES_BEFORE'
RETURN a.label as source_table, 
       b.label as target_table,
       r.properties_json.sequence_order as order,
       r.properties_json.sequence_type as sequence_type
ORDER BY r.properties_json.sequence_order`
}

// QueryTableClassifications returns a Cypher query to find all table classifications.
func (gqh *GraphQueryHelpers) QueryTableClassifications() string {
	return `MATCH (n:Node)
WHERE n.type = 'table'
  AND EXISTS(n.properties_json.table_classification)
RETURN n.label as table_name,
       n.properties_json.table_classification as classification,
       n.properties_json.classification_confidence as confidence
ORDER BY n.label`
}

// QueryCodeParameters returns a Cypher query to find code parameters.
func (gqh *GraphQueryHelpers) QueryCodeParameters(sourceType string) string {
	if sourceType != "" {
		return fmt.Sprintf(`MATCH (n:Node)
WHERE n.type = 'parameter'
  AND n.properties_json.source = '%s'
RETURN n.label as parameter_name,
       n.properties_json.type as parameter_type,
       n.properties_json.context as context,
       n.properties_json.is_required as is_required
ORDER BY n.label`, sourceType)
	}
	return `MATCH (n:Node)
WHERE n.type = 'parameter'
RETURN n.label as parameter_name,
       n.properties_json.type as parameter_type,
       n.properties_json.source as source,
       n.properties_json.context as context,
       n.properties_json.is_required as is_required
ORDER BY n.properties_json.source, n.label`
}

// QueryHardcodedLists returns a Cypher query to find hardcoded lists.
func (gqh *GraphQueryHelpers) QueryHardcodedLists() string {
	return `MATCH (n:Node)
WHERE n.type = 'hardcoded_list'
RETURN n.label as list_name,
       n.properties_json.values as values,
       n.properties_json.type as list_type,
       n.properties_json.context as context
ORDER BY n.label`
}

// QueryTestingEndpoints returns a Cypher query to find testing endpoints.
func (gqh *GraphQueryHelpers) QueryTestingEndpoints() string {
	return `MATCH (n:Node)
WHERE n.type = 'endpoint'
  AND n.properties_json.is_test = true
RETURN n.label as endpoint,
       n.properties_json.method as method,
       n.properties_json.source as source,
       n.properties_json.test_indicators as indicators
ORDER BY n.label`
}

// QueryAdvancedExtractionSummary returns a Cypher query to get summary of advanced extraction results.
func (gqh *GraphQueryHelpers) QueryAdvancedExtractionSummary() string {
	return `MATCH (n:Node)
WHERE n.type IN ['table', 'parameter', 'hardcoded_list', 'endpoint']
WITH n.type as node_type, count(*) as count
RETURN node_type, count
ORDER BY node_type`
}

