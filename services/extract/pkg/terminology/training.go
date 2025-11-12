package terminology

import (
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// BatchTerminologyTrainer performs periodic batch refinement from knowledge graph.
type BatchTerminologyTrainer struct {
	terminologyLearner *TerminologyLearner
	neo4jPersistence   *storage.Neo4jPersistence
	logger             *log.Logger
	trainingInterval   time.Duration
	lastTraining       time.Time
}

// NewBatchTerminologyTrainer creates a new batch terminology trainer.
func NewBatchTerminologyTrainer(
	terminologyLearner *TerminologyLearner,
	neo4jPersistence *storage.Neo4jPersistence,
	logger *log.Logger,
	trainingInterval time.Duration,
) *BatchTerminologyTrainer {
	return &BatchTerminologyTrainer{
		terminologyLearner: terminologyLearner,
		neo4jPersistence:   neo4jPersistence,
		logger:             logger,
		trainingInterval:   trainingInterval,
		lastTraining:       time.Now(),
	}
}

// TrainFromKnowledgeGraph performs batch training from accumulated knowledge graph data.
func (btt *BatchTerminologyTrainer) TrainFromKnowledgeGraph(ctx context.Context) error {
	btt.logger.Println("Starting batch terminology training from knowledge graph...")

	// Query knowledge graph for terminology examples
	query := `
		MATCH (n:graph.Node)
		WHERE n.label IS NOT NULL
		RETURN n.id AS id, n.label AS label, n.type AS type, n.properties_json AS props
		ORDER BY n.updated_at DESC
		LIMIT 10000
	`

	session := btt.neo4jPersistence.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return err
	}

	// Convert Neo4j results to nodes
	records, ok := result.([]*neo4j.Record)
	if !ok {
		btt.logger.Println("No nodes found in knowledge graph for training")
		return nil
	}

	nodes := []graph.Node{}
	for _, record := range records {
		id, _ := record.Get("id")
		label, _ := record.Get("label")
		nodeType, _ := record.Get("type")
		propsJSON, _ := record.Get("props")

		node := graph.Node{
			ID:    getString(id),
			Label: getString(label),
			Type:  getString(nodeType),
			Props: parsePropsJSON(getString(propsJSON)),
		}
		nodes = append(nodes, node)
	}

	// Learn from nodes
	if err := btt.terminologyLearner.LearnFromExtraction(ctx, nodes, []Edge{}); err != nil {
		return err
	}

	btt.lastTraining = time.Now()
	btt.logger.Printf("Batch terminology training completed. Processed %d nodes", len(nodes))

	return nil
}

// ShouldTrain returns true if it's time for batch training.
func (btt *BatchTerminologyTrainer) ShouldTrain() bool {
	return time.Since(btt.lastTraining) >= btt.trainingInterval
}

// Helper functions

func getString(v any) string {
	if str, ok := v.(string); ok {
		return str
	}
	return ""
}

func parsePropsJSON(jsonStr string) map[string]any {
	if jsonStr == "" {
		return nil
	}
	var props map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &props); err != nil {
		return nil
	}
	return props
}

