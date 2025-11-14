package embeddings

import (
	"github.com/plturrell/aModels/services/extract/pkg/graph"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sync"
)

// BatchEmbeddingRequest represents a batch embedding request
type BatchEmbeddingRequest struct {
	ArtifactType string
	Items        []BatchItem
}

// BatchItem represents a single item in a batch
type BatchItem struct {
	ID       string
	Data     interface{}
	Metadata map[string]any
}

// BatchEmbeddingResult represents a batch embedding result
type BatchEmbeddingResult struct {
	ID         string
	Relational []float32
	Semantic   []float32
	Embedding  []float32
	Error      error
}

// BatchEmbeddingGenerator handles batch embedding generation
type BatchEmbeddingGenerator struct {
	logger      *log.Logger
	cache       *EmbeddingCache
	batchSize   int
	useSemantic bool
}

// NewBatchEmbeddingGenerator creates a new batch embedding generator
func NewBatchEmbeddingGenerator(logger *log.Logger, cache *EmbeddingCache, batchSize int) *BatchEmbeddingGenerator {
	return &BatchEmbeddingGenerator{
		logger:      logger,
		cache:       cache,
		batchSize:   batchSize,
		useSemantic: os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true",
	}
}

// GenerateBatchTableEmbeddings generates embeddings for multiple tables in batch
func (beg *BatchEmbeddingGenerator) GenerateBatchTableEmbeddings(
	ctx context.Context,
	nodes []graph.Node,
) (map[string]BatchEmbeddingResult, error) {
	results := make(map[string]BatchEmbeddingResult)
	
	// Check cache first
	uncached := []BatchItem{}
	for _, node := range nodes {
		relational, semantic, _, cached := beg.cache.Get("table", node)
		if cached {
			results[node.ID] = BatchEmbeddingResult{
				ID:         node.ID,
				Relational: relational,
				Semantic:   semantic,
			}
		} else {
			uncached = append(uncached, BatchItem{
				ID:       node.ID,
				Data:     node,
				Metadata: node.Props,
			})
		}
	}

	if len(uncached) == 0 {
		return results, nil
	}

	// Process in batches
	for i := 0; i < len(uncached); i += beg.batchSize {
		end := i + beg.batchSize
		if end > len(uncached) {
			end = len(uncached)
		}

		batch := uncached[i:end]
		batchResults := beg.processBatchTableEmbeddings(ctx, batch)
		
		// Store results and cache
		for _, result := range batchResults {
			results[result.ID] = result
			
			// Cache successful results
			if result.Error == nil {
				// Find the node for this result
				var node graph.Node
				for _, item := range batch {
					if item.ID == result.ID {
						if n, ok := item.Data.(graph.Node); ok {
							node = n
							break
						}
					}
				}
				
				if node.ID != "" {
					beg.cache.Set("table", node, result.Relational, result.Semantic, nil, node.Props)
				}
			}
		}
	}

	return results, nil
}

// processBatchTableEmbeddings processes a batch of table embeddings
func (beg *BatchEmbeddingGenerator) processBatchTableEmbeddings(
	ctx context.Context,
	batch []BatchItem,
) []BatchEmbeddingResult {
	results := make([]BatchEmbeddingResult, len(batch))
	
	// Process relational embeddings in parallel
	var wg sync.WaitGroup
	for i, item := range batch {
		wg.Add(1)
		go func(idx int, it BatchItem) {
			defer wg.Done()
			
			node, ok := it.Data.(graph.Node)
			if !ok {
				results[idx].Error = fmt.Errorf("invalid node data")
				return
			}
			
			results[idx].ID = it.ID
			
			// Generate relational embedding
			relational, err := beg.generateRelationalTableEmbedding(ctx, node)
			if err != nil {
				results[idx].Error = err
				return
			}
			results[idx].Relational = relational
			
			// Generate semantic embedding if enabled
			if beg.useSemantic {
				semantic, err := beg.generateSemanticTableEmbedding(ctx, node)
				if err != nil {
					beg.logger.Printf("semantic embedding failed for %s: %v", node.Label, err)
					// Non-fatal, continue with relational
				} else {
					results[idx].Semantic = semantic
				}
			}
		}(i, item)
	}
	
	wg.Wait()
	
	return results
}

// generateRelationalTableEmbedding generates relational embedding for a table
func (beg *BatchEmbeddingGenerator) generateRelationalTableEmbedding(ctx context.Context, node graph.Node) ([]float32, error) {
	columns := []map[string]any{}
	if node.Props != nil {
		if cols, ok := node.Props["columns"].([]map[string]any); ok {
			columns = cols
		}
	}

	if len(columns) == 0 {
		columns = []map[string]any{
			{"name": node.Label, "type": "string"},
		}
	}

	columnsJSON, err := json.Marshal(columns)
	if err != nil {
		return nil, fmt.Errorf("marshal columns: %w", err)
	}

	metadataJSON := "{}"
	if node.Props != nil {
		metadataBytes, err := json.Marshal(node.Props)
		if err == nil {
			metadataJSON = string(metadataBytes)
		}
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
		"--artifact-type", "table",
		"--table-name", node.Label,
		"--columns", string(columnsJSON),
		"--metadata", metadataJSON,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("generate table embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("generate table embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("unmarshal embedding: %w", err)
	}

	return embedding, nil
}

// generateSemanticTableEmbedding generates semantic embedding for a table
func (beg *BatchEmbeddingGenerator) generateSemanticTableEmbedding(ctx context.Context, node graph.Node) ([]float32, error) {
	columns := []map[string]any{}
	if node.Props != nil {
		if cols, ok := node.Props["columns"].([]map[string]any); ok {
			columns = cols
		}
	}

	columnsJSON, err := json.Marshal(columns)
	if err != nil {
		return nil, fmt.Errorf("marshal columns: %w", err)
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed_sap_rpt.py",
		"--artifact-type", "table",
		"--table-name", node.Label,
		"--columns", string(columnsJSON),
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("generate semantic embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("generate semantic embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("unmarshal semantic embedding: %w", err)
	}

	return embedding, nil
}

// GenerateBatchColumnEmbeddings generates embeddings for multiple columns in batch
func (beg *BatchEmbeddingGenerator) GenerateBatchColumnEmbeddings(
	ctx context.Context,
	nodes []graph.Node,
) (map[string]BatchEmbeddingResult, error) {
	results := make(map[string]BatchEmbeddingResult)
	
	// Check cache first
	uncached := []BatchItem{}
	for _, node := range nodes {
		embedding, _, _, cached := beg.cache.Get("column", node)
		if cached {
			results[node.ID] = BatchEmbeddingResult{
				ID:        node.ID,
				Embedding: embedding,
			}
		} else {
			uncached = append(uncached, BatchItem{
				ID:       node.ID,
				Data:     node,
				Metadata: node.Props,
			})
		}
	}

	if len(uncached) == 0 {
		return results, nil
	}

	// Process in batches with parallel execution
	var wg sync.WaitGroup
	resultChan := make(chan BatchEmbeddingResult, len(uncached))

	for i := 0; i < len(uncached); i += beg.batchSize {
		end := i + beg.batchSize
		if end > len(uncached) {
			end = len(uncached)
		}

		batch := uncached[i:end]
		for _, item := range batch {
			wg.Add(1)
			go func(it BatchItem) {
				defer wg.Done()
				
				node, ok := it.Data.(graph.Node)
				if !ok {
					resultChan <- BatchEmbeddingResult{
						ID:    it.ID,
						Error: fmt.Errorf("invalid node data"),
					}
					return
				}
				
				embedding, err := GenerateColumnEmbedding(ctx, node)
				if err != nil {
					resultChan <- BatchEmbeddingResult{
						ID:    it.ID,
						Error: err,
					}
					return
				}
				
				// Cache result
				beg.cache.Set("column", node, nil, nil, embedding, node.Props)
				
				resultChan <- BatchEmbeddingResult{
					ID:        it.ID,
					Embedding: embedding,
				}
			}(item)
		}
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for result := range resultChan {
		results[result.ID] = result
	}

	return results, nil
}

