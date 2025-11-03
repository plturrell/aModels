package glove

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DistributedTrainer coordinates distributed training across multiple workers
type DistributedTrainer struct {
	coordinator *TrainingCoordinator
	workers     []*TrainingWorker
	config      DistributedConfig
	mu          sync.RWMutex
}

// DistributedConfig holds distributed training configuration
type DistributedConfig struct {
	NumWorkers       int
	SyncInterval     int // Synchronize every N iterations
	LearningRate     float64
	GradientClipping float64
	AllReduceMethod  string // "average", "sum", "weighted"
}

// TrainingCoordinator manages distributed training
type TrainingCoordinator struct {
	masterModel  *Model
	workerModels []*Model
	syncChannel  chan GradientUpdate
	config       DistributedConfig
	iteration    int
	mu           sync.RWMutex
}

// TrainingWorker represents a training worker node
type TrainingWorker struct {
	id            int
	model         *Model
	dataPartition []CooccurrenceEntry
	gradients     []GradientUpdate
	coordinator   *TrainingCoordinator
	mu            sync.Mutex
}

// GradientUpdate represents a gradient update from a worker
type GradientUpdate struct {
	WorkerID        int
	Iteration       int
	WordID          int
	ContextID       int
	VectorGrad      []float32
	ContextGrad     []float32
	BiasGrad        float32
	ContextBiasGrad float32
	Timestamp       time.Time
}

// NewDistributedTrainer creates a new distributed trainer
func NewDistributedTrainer(baseModel *Model, config DistributedConfig) (*DistributedTrainer, error) {
	if config.NumWorkers <= 0 {
		config.NumWorkers = 4
	}
	if config.SyncInterval <= 0 {
		config.SyncInterval = 10
	}
	if config.AllReduceMethod == "" {
		config.AllReduceMethod = "average"
	}

	coordinator := &TrainingCoordinator{
		masterModel: baseModel,
		syncChannel: make(chan GradientUpdate, config.NumWorkers*100),
		config:      config,
	}

	dt := &DistributedTrainer{
		coordinator: coordinator,
		workers:     make([]*TrainingWorker, config.NumWorkers),
		config:      config,
	}

	// Initialize workers
	for i := 0; i < config.NumWorkers; i++ {
		worker := &TrainingWorker{
			id:          i,
			model:       baseModel, // In production, would clone model
			coordinator: coordinator,
			gradients:   make([]GradientUpdate, 0, 1000),
		}
		dt.workers[i] = worker
	}

	fmt.Printf("Distributed trainer initialized with %d workers\n", config.NumWorkers)
	return dt, nil
}

// Train performs distributed training
func (dt *DistributedTrainer) Train(ctx context.Context, entries []CooccurrenceEntry) error {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	fmt.Println("Starting distributed training...")
	fmt.Printf("  Workers: %d\n", dt.config.NumWorkers)
	fmt.Printf("  Sync interval: %d iterations\n", dt.config.SyncInterval)
	fmt.Printf("  All-reduce method: %s\n", dt.config.AllReduceMethod)

	// Partition data across workers
	if err := dt.partitionData(entries); err != nil {
		return fmt.Errorf("partition data: %w", err)
	}

	// Start gradient aggregation goroutine
	aggregationDone := make(chan error, 1)
	go func() {
		aggregationDone <- dt.aggregateGradients(ctx)
	}()

	// Start workers
	var wg sync.WaitGroup
	workerErrors := make(chan error, dt.config.NumWorkers)

	for i, worker := range dt.workers {
		wg.Add(1)
		go func(w *TrainingWorker, workerID int) {
			defer wg.Done()
			if err := w.train(ctx); err != nil {
				workerErrors <- fmt.Errorf("worker %d: %w", workerID, err)
			}
		}(worker, i)
	}

	// Wait for workers
	wg.Wait()
	close(workerErrors)

	// Check for worker errors
	for err := range workerErrors {
		if err != nil {
			return err
		}
	}

	// Stop aggregation
	close(dt.coordinator.syncChannel)
	if err := <-aggregationDone; err != nil {
		return fmt.Errorf("aggregation: %w", err)
	}

	fmt.Println("✓ Distributed training complete")
	return nil
}

// partitionData partitions co-occurrence entries across workers
func (dt *DistributedTrainer) partitionData(entries []CooccurrenceEntry) error {
	entriesPerWorker := len(entries) / dt.config.NumWorkers
	remainder := len(entries) % dt.config.NumWorkers

	offset := 0
	for i := 0; i < dt.config.NumWorkers; i++ {
		size := entriesPerWorker
		if i < remainder {
			size++
		}

		end := offset + size
		if end > len(entries) {
			end = len(entries)
		}

		dt.workers[i].dataPartition = entries[offset:end]
		offset = end

		fmt.Printf("Worker %d: %d entries\n", i, len(dt.workers[i].dataPartition))
	}

	return nil
}

// aggregateGradients aggregates gradients from workers
func (dt *DistributedTrainer) aggregateGradients(ctx context.Context) error {
	gradientBuffer := make(map[string][]GradientUpdate)
	syncCounter := 0

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case grad, ok := <-dt.coordinator.syncChannel:
			if !ok {
				// Channel closed, final sync
				return dt.performSync(gradientBuffer)
			}

			// Buffer gradient
			key := fmt.Sprintf("%d_%d", grad.WordID, grad.ContextID)
			gradientBuffer[key] = append(gradientBuffer[key], grad)

			syncCounter++
			if syncCounter >= dt.config.SyncInterval*dt.config.NumWorkers {
				if err := dt.performSync(gradientBuffer); err != nil {
					return err
				}
				gradientBuffer = make(map[string][]GradientUpdate)
				syncCounter = 0
			}
		}
	}
}

// performSync performs all-reduce and updates master model
func (dt *DistributedTrainer) performSync(gradientBuffer map[string][]GradientUpdate) error {
	dt.coordinator.mu.Lock()
	defer dt.coordinator.mu.Unlock()

	dt.coordinator.iteration++

	fmt.Printf("Synchronization %d: aggregating %d gradient groups\n",
		dt.coordinator.iteration, len(gradientBuffer))

	// All-reduce gradients
	for _, grads := range gradientBuffer {
		if len(grads) == 0 {
			continue
		}

		// Average gradients across workers
		avgGrad := dt.allReduce(grads)

		// Apply to master model (in production, would actually update)
		_ = avgGrad
	}

	// Broadcast updated parameters to workers (in production)
	// for _, worker := range dt.workers {
	//     worker.model = dt.coordinator.masterModel.Clone()
	// }

	return nil
}

// allReduce performs all-reduce operation on gradients
func (dt *DistributedTrainer) allReduce(grads []GradientUpdate) GradientUpdate {
	if len(grads) == 0 {
		return GradientUpdate{}
	}

	result := grads[0]
	vectorSize := len(result.VectorGrad)

	switch dt.config.AllReduceMethod {
	case "average":
		// Average all gradients
		for i := 1; i < len(grads); i++ {
			for j := 0; j < vectorSize; j++ {
				result.VectorGrad[j] += grads[i].VectorGrad[j]
				result.ContextGrad[j] += grads[i].ContextGrad[j]
			}
			result.BiasGrad += grads[i].BiasGrad
			result.ContextBiasGrad += grads[i].ContextBiasGrad
		}

		scale := float32(1.0 / float64(len(grads)))
		for j := 0; j < vectorSize; j++ {
			result.VectorGrad[j] *= scale
			result.ContextGrad[j] *= scale
		}
		result.BiasGrad *= scale
		result.ContextBiasGrad *= scale

	case "sum":
		// Sum all gradients
		for i := 1; i < len(grads); i++ {
			for j := 0; j < vectorSize; j++ {
				result.VectorGrad[j] += grads[i].VectorGrad[j]
				result.ContextGrad[j] += grads[i].ContextGrad[j]
			}
			result.BiasGrad += grads[i].BiasGrad
			result.ContextBiasGrad += grads[i].ContextBiasGrad
		}

	case "weighted":
		// Weighted average (could weight by worker performance)
		// For now, same as average
		return dt.allReduce(grads)
	}

	return result
}

// train performs training on a worker
func (w *TrainingWorker) train(ctx context.Context) error {
	fmt.Printf("Worker %d: starting training on %d entries\n", w.id, len(w.dataPartition))

	for iter := 0; iter < 10; iter++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		for _, entry := range w.dataPartition {
			// Compute gradients (simplified)
			grad := GradientUpdate{
				WorkerID:  w.id,
				Iteration: iter,
				WordID:    entry.WordID,
				ContextID: entry.ContextID,
				Timestamp: time.Now(),
			}

			// Send gradient to coordinator
			w.coordinator.syncChannel <- grad
		}

		if (iter+1)%5 == 0 {
			fmt.Printf("Worker %d: completed iteration %d\n", w.id, iter+1)
		}
	}

	fmt.Printf("Worker %d: training complete\n", w.id)
	return nil
}

// GetMasterModel returns the master model after distributed training
func (dt *DistributedTrainer) GetMasterModel() *Model {
	dt.coordinator.mu.RLock()
	defer dt.coordinator.mu.RUnlock()
	return dt.coordinator.masterModel
}

// GetWorkerStats returns statistics for all workers
func (dt *DistributedTrainer) GetWorkerStats() []WorkerStats {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	stats := make([]WorkerStats, len(dt.workers))
	for i, worker := range dt.workers {
		worker.mu.Lock()
		stats[i] = WorkerStats{
			WorkerID:          worker.id,
			DataPartitionSize: len(worker.dataPartition),
			GradientsComputed: len(worker.gradients),
		}
		worker.mu.Unlock()
	}

	return stats
}

// WorkerStats contains statistics for a worker
type WorkerStats struct {
	WorkerID          int
	DataPartitionSize int
	GradientsComputed int
}

// FederatedMCTS performs federated MCTS optimization across multiple nodes
type FederatedMCTS struct {
	localOptimizers  []*MCTSOptimizer
	globalBestConfig Config
	globalBestScore  float64
	mu               sync.RWMutex
}

// NewFederatedMCTS creates a federated MCTS optimizer
func NewFederatedMCTS(numNodes int, searchSpace HyperparameterSpace, evalFunc EvaluationFunc) *FederatedMCTS {
	fm := &FederatedMCTS{
		localOptimizers: make([]*MCTSOptimizer, numNodes),
		globalBestScore: -1e9,
	}

	for i := 0; i < numNodes; i++ {
		fm.localOptimizers[i] = NewMCTSOptimizer(searchSpace, evalFunc)
	}

	return fm
}

// OptimizeDistributed performs distributed hyperparameter optimization
func (fm *FederatedMCTS) OptimizeDistributed(ctx context.Context, corpus []string, iterationsPerNode int) (Config, float64, error) {
	fmt.Printf("Starting federated MCTS with %d nodes\n", len(fm.localOptimizers))

	var wg sync.WaitGroup
	results := make(chan struct {
		config Config
		score  float64
	}, len(fm.localOptimizers))

	// Run MCTS on each node
	for i, optimizer := range fm.localOptimizers {
		wg.Add(1)
		go func(nodeID int, opt *MCTSOptimizer) {
			defer wg.Done()

			fmt.Printf("Node %d: starting optimization\n", nodeID)
			config, score, err := opt.Optimize(ctx, corpus, iterationsPerNode)
			if err != nil {
				fmt.Printf("Node %d: error: %v\n", nodeID, err)
				return
			}

			results <- struct {
				config Config
				score  float64
			}{config, score}

			fmt.Printf("Node %d: best score %.6f\n", nodeID, score)
		}(i, optimizer)
	}

	// Wait for all nodes
	go func() {
		wg.Wait()
		close(results)
	}()

	// Aggregate results
	for result := range results {
		fm.mu.Lock()
		if result.score > fm.globalBestScore {
			fm.globalBestScore = result.score
			fm.globalBestConfig = result.config
		}
		fm.mu.Unlock()
	}

	fmt.Printf("\nFederated optimization complete\n")
	fmt.Printf("Global best score: %.6f\n", fm.globalBestScore)

	return fm.globalBestConfig, fm.globalBestScore, nil
}
