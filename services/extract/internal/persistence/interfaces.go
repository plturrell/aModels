package persistence

import "errors"

// Node represents a graph node (matches main package structure)
type Node struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Label string         `json:"label,omitempty"`
	Props map[string]any `json:"props,omitempty"`
}

// Edge represents a graph edge (matches main package structure)
type Edge struct {
	SourceID string         `json:"source_id"`
	TargetID string         `json:"target_id"`
	Label    string         `json:"label,omitempty"`
	Props    map[string]any `json:"props,omitempty"`
}

// GraphPersistence handles saving graph data (nodes and edges)
type GraphPersistence interface {
	SaveGraph(nodes []Node, edges []Edge) error
}

// TablePersistence handles saving table data
type TablePersistence interface {
	SaveTable(tableName string, data []map[string]any) error
}

// DocumentPersistence handles saving document files
type DocumentPersistence interface {
	SaveDocument(path string) error
}

// VectorPersistence handles saving vector embeddings
type VectorPersistence interface {
	SaveVector(key string, vector []float32) error
}

// CompositeGraphPersistence combines multiple graph persistence implementations
type CompositeGraphPersistence struct {
	stores []GraphPersistence
}

// NewCompositeGraphPersistence creates a composite graph persistence
func NewCompositeGraphPersistence(stores ...GraphPersistence) GraphPersistence {
	if len(stores) == 0 {
		return nil
	}
	if len(stores) == 1 {
		return stores[0]
	}
	return &CompositeGraphPersistence{stores: append([]GraphPersistence(nil), stores...)}
}

// SaveGraph saves graph data to all underlying stores
func (c *CompositeGraphPersistence) SaveGraph(nodes []Node, edges []Edge) error {
	var errs []error
	for _, store := range c.stores {
		if store == nil {
			continue
		}
		if err := store.SaveGraph(nodes, edges); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

