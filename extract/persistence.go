package main

import (
	"errors"
	_ "github.com/lib/pq"
	_ "github.com/mattn/go-sqlite3"
	_ "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	_ "github.com/redis/go-redis/v9"
)

type GraphPersistence interface {
	SaveGraph(nodes []Node, edges []Edge) error
}

type compositeGraphPersistence struct {
	stores []GraphPersistence
}

func newCompositeGraphPersistence(stores ...GraphPersistence) GraphPersistence {
	if len(stores) == 0 {
		return nil
	}
	if len(stores) == 1 {
		return stores[0]
	}
	return compositeGraphPersistence{stores: append([]GraphPersistence(nil), stores...)}
}

func (c compositeGraphPersistence) SaveGraph(nodes []Node, edges []Edge) error {
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

type TablePersistence interface {
	SaveTable(tableName string, data []map[string]any) error
}

type DocumentPersistence interface {
	SaveDocument(path string) error
}

type VectorPersistence interface {
	SaveVector(key string, vector []float32) error
}
