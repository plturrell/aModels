package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// GleanPersistence exports graph facts into JSON batches compatible with the
// `glean write` CLI. Users can ingest these batches into a Glean database to
// query ETL metadata alongside other indexed assets.
type GleanPersistence struct {
	exportDir           string
	predicatePrefix     string
	logger              *log.Logger
	mu                  sync.Mutex
	sequence            uint64
	includeEmptyBatches bool
	schemaCheck         sync.Once
	schemaPath          string
}

const (
	defaultGleanPredicatePrefix = "agenticAiETH.ETL"

	nodePredicateName     = "Node"
	edgePredicateName     = "Edge"
	manifestPredicateName = "ExportManifest"
)

// NewGleanPersistence initialises a persistence that writes batches to exportDir.
func NewGleanPersistence(exportDir, predicatePrefix string, logger *log.Logger) (*GleanPersistence, error) {
	if predicatePrefix == "" {
		predicatePrefix = defaultGleanPredicatePrefix
	}

	if exportDir == "" {
		return nil, fmt.Errorf("glean export directory is required")
	}
	absDir, err := filepath.Abs(exportDir)
	if err != nil {
		return nil, fmt.Errorf("resolve glean export directory: %w", err)
	}
	if err := os.MkdirAll(absDir, 0o755); err != nil {
		return nil, fmt.Errorf("create glean export directory: %w", err)
	}

	if logger == nil {
		logger = log.New(os.Stdout, "[glean] ", log.LstdFlags|log.Lmsgprefix)
	}

	schemaPath := strings.TrimSpace(os.Getenv("GLEAN_SCHEMA_PATH"))
	if schemaPath == "" {
		schemaRoot := strings.TrimSpace(os.Getenv("GLEAN_SCHEMA_ROOT"))
		if schemaRoot != "" {
			schemaPath = filepath.Join(schemaRoot, "source", "etl.angle")
		} else {
			schemaPath = filepath.Join("glean", "schema", "source", "etl.angle")
		}
	}

	return &GleanPersistence{
		exportDir:       absDir,
		predicatePrefix: predicatePrefix,
		logger:          logger,
		schemaPath:      schemaPath,
	}, nil
}

// SaveGraph serialises the supplied nodes and edges into a JSON batch file in
// the configured export directory. The resulting file can be ingested with:
//
//	glean write --schema <schema-path> --db <name/instance> <batch-file>
//
// followed by `glean finish`.
func (g *GleanPersistence) SaveGraph(nodes []Node, edges []Edge) error {
	g.warnIfSchemaMissing()
	if len(nodes) == 0 && len(edges) == 0 && !g.includeEmptyBatches {
		return nil
	}

	uniqueNodes := dedupeNodes(nodes)
	uniqueEdges := dedupeEdges(edges)

	batch, err := g.buildBatch(uniqueNodes, uniqueEdges)
	if err != nil {
		return err
	}
	if len(batch) == 0 && !g.includeEmptyBatches {
		return nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	timestamp := time.Now().UTC()
	filename := fmt.Sprintf("graph_%s_%04d.json", timestamp.Format("20060102T150405Z"), g.sequence)
	g.sequence++

	if len(batch) > 0 {
		exportSuffix := g.predicate(manifestPredicateName)
		for i := range batch {
			if batch[i].Predicate != exportSuffix || len(batch[i].Facts) == 0 {
				continue
			}
			if meta, ok := batch[i].Facts[0].Key.(map[string]any); ok {
				meta["export_file"] = filename
			}
		}
	}

	filePath := filepath.Join(g.exportDir, filename)
	if err := writeGleanBatch(filePath, batch); err != nil {
		return err
	}

	if g.logger != nil {
		g.logger.Printf("glean export wrote %s (%d nodes, %d edges)", filePath, len(uniqueNodes), len(uniqueEdges))
	}
	return nil
}

func (g *GleanPersistence) warnIfSchemaMissing() {
	g.schemaCheck.Do(func() {
		if g.schemaPath == "" {
			return
		}
		if _, err := os.Stat(g.schemaPath); err != nil {
			if g.logger != nil {
				g.logger.Printf("warning: glean schema file not found at %s; ensure the Angle schema is compiled before ingest", g.schemaPath)
			}
		}
	})
}

func (g *GleanPersistence) buildBatch(nodes []Node, edges []Edge) ([]gleanPredicate, error) {
	nodeFacts := make([]gleanFact, 0, len(nodes))
	for _, node := range nodes {
		if node.ID == "" {
			continue
		}
		key := map[string]any{
			"id":    node.ID,
			"kind":  node.Type,
			"label": node.Label,
		}
		if len(node.Props) > 0 {
			propsJSON, err := encodeProperties(node.Props)
			if err != nil {
				return nil, fmt.Errorf("encode node properties for %q: %w", node.ID, err)
			}
			key["properties_json"] = propsJSON
		}
		nodeFacts = append(nodeFacts, gleanFact{Key: key})
	}

	edgeFacts := make([]gleanFact, 0, len(edges))
	for _, edge := range edges {
		if edge.SourceID == "" || edge.TargetID == "" {
			continue
		}
		key := map[string]any{
			"source": edge.SourceID,
			"target": edge.TargetID,
			"label":  edge.Label,
		}
		if len(edge.Props) > 0 {
			propsJSON, err := encodeProperties(edge.Props)
			if err != nil {
				return nil, fmt.Errorf("encode edge properties for %s->%s: %w", edge.SourceID, edge.TargetID, err)
			}
			key["properties_json"] = propsJSON
		}
		edgeFacts = append(edgeFacts, gleanFact{Key: key})
	}

	predicates := make([]gleanPredicate, 0, 3)
	if len(nodeFacts) > 0 {
		predicates = append(predicates, gleanPredicate{
			Predicate: g.predicate(nodePredicateName),
			Facts:     nodeFacts,
		})
	}
	if len(edgeFacts) > 0 {
		predicates = append(predicates, gleanPredicate{
			Predicate: g.predicate(edgePredicateName),
			Facts:     edgeFacts,
		})
	}

	exportMeta := gleanFact{
		Key: map[string]any{
			"exported_at": time.Now().UTC().Format(time.RFC3339Nano),
			"node_count":  len(nodeFacts),
			"edge_count":  len(edgeFacts),
		},
	}
	predicates = append(predicates, gleanPredicate{
		Predicate: g.predicate(manifestPredicateName),
		Facts:     []gleanFact{exportMeta},
	})

	return predicates, nil
}

func (g *GleanPersistence) predicate(name string) string {
	return fmt.Sprintf("%s.%s.1", g.predicatePrefix, name)
}

// PredicatePrefix exposes the configured predicate namespace.
func (g *GleanPersistence) PredicatePrefix() string {
	return g.predicatePrefix
}

// ExportDir returns the absolute export directory.
func (g *GleanPersistence) ExportDir() string {
	return g.exportDir
}

func writeGleanBatch(path string, batch []gleanPredicate) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create glean batch %s: %w", path, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(batch); err != nil {
		return fmt.Errorf("encode glean batch: %w", err)
	}
	return nil
}

func encodeProperties(props map[string]any) (string, error) {
	data, err := json.Marshal(props)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func dedupeNodes(nodes []Node) []Node {
	if len(nodes) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(nodes))
	result := make([]Node, 0, len(nodes))
	for _, node := range nodes {
		if node.ID == "" {
			continue
		}
		if _, ok := seen[node.ID]; ok {
			continue
		}
		seen[node.ID] = struct{}{}
		result = append(result, node)
	}
	return result
}

func dedupeEdges(edges []Edge) []Edge {
	if len(edges) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(edges))
	result := make([]Edge, 0, len(edges))
	for _, edge := range edges {
		if edge.SourceID == "" || edge.TargetID == "" {
			continue
		}
		key := edge.SourceID + "->" + edge.TargetID + "#" + edge.Label
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, edge)
	}
	return result
}

type gleanPredicate struct {
	Predicate string      `json:"predicate"`
	Facts     []gleanFact `json:"facts"`
}

type gleanFact struct {
	Key   any `json:"key"`
	Value any `json:"value,omitempty"`
}
