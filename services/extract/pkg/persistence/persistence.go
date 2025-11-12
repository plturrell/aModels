package persistence

import (
	"errors"
	"fmt"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
	_ "github.com/lib/pq"
	_ "github.com/mattn/go-sqlite3"
	_ "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	_ "github.com/redis/go-redis/v9"
)

type GraphPersistence interface {
	SaveGraph(nodes []graph.Node, edges []graph.Edge) error
}

type compositeGraphPersistence struct {
	stores []GraphPersistence
}

func NewCompositeGraphPersistence(stores ...GraphPersistence) GraphPersistence {
	if len(stores) == 0 {
		return nil
	}
	if len(stores) == 1 {
		return stores[0]
	}
	return compositeGraphPersistence{stores: append([]GraphPersistence(nil), stores...)}
}

func (c compositeGraphPersistence) SaveGraph(nodes []graph.Node, edges []graph.Edge) error {
	var errs []error
	for i, store := range c.stores {
		if store == nil {
			continue
		}
		if err := store.SaveGraph(nodes, edges); err != nil {
			errs = append(errs, fmt.Errorf("store %d failed: %w", i, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("composite persistence errors: %w", errors.Join(errs...))
	}
	return nil
}

type TablePersistence interface {
	SaveTable(tableName string, data []map[string]any) error
}

type DocumentPersistence interface {
	SaveDocument(path string) error
}

// VectorSearchResult represents a result from vector similarity search
type VectorSearchResult struct {
	Key          string
	ArtifactType string
	ArtifactID   string
	Vector       []float32
	Metadata     map[string]any
	Score        float32
	Text         string
}

type VectorPersistence interface {
	SaveVector(key string, vector []float32, metadata map[string]any) error
	GetVector(key string) ([]float32, map[string]any, error)
	SearchSimilar(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error)
	SearchByText(query string, artifactType string, limit int) ([]VectorSearchResult, error)
}
