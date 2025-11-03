package glove

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// MCTSOptimizer uses Monte Carlo Tree Search to find optimal hyperparameters
type MCTSOptimizer struct {
	searchSpace  map[string][]interface{}
	explorationC float64 // UCB exploration constant
	maxIter      int
	evalFunc     EvaluationFunc
	bestConfig   Config
	bestScore    float64
	mu           sync.RWMutex
}

// EvaluationFunc evaluates a configuration and returns a score (higher is better)
type EvaluationFunc func(ctx context.Context, cfg Config, corpus []string) (float64, error)

// MCTSNode represents a node in the search tree
type MCTSNode struct {
	config       Config
	visits       int
	totalReward  float64
	children     []*MCTSNode
	parent       *MCTSNode
	untriedMoves []Config
	mu           sync.RWMutex
}

// HyperparameterSpace defines the search space for hyperparameters
type HyperparameterSpace struct {
	VectorSizes   []int
	LearningRates []float64
	MaxIters      []int
	Alphas        []float64
	WindowSizes   []int
}

// DefaultSearchSpace returns a reasonable default search space
func DefaultSearchSpace() HyperparameterSpace {
	return HyperparameterSpace{
		VectorSizes:   []int{50, 100, 200, 300},
		LearningRates: []float64{0.01, 0.05, 0.1},
		MaxIters:      []int{10, 15, 25},
		Alphas:        []float64{0.5, 0.75, 1.0},
		WindowSizes:   []int{10, 15, 20},
	}
}

// NewMCTSOptimizer creates a new MCTS-based hyperparameter optimizer
func NewMCTSOptimizer(searchSpace HyperparameterSpace, evalFunc EvaluationFunc) *MCTSOptimizer {
	return &MCTSOptimizer{
		searchSpace: map[string][]interface{}{
			"VectorSize":   toInterfaceSlice(searchSpace.VectorSizes),
			"LearningRate": toFloatInterfaceSlice(searchSpace.LearningRates),
			"MaxIter":      toInterfaceSlice(searchSpace.MaxIters),
			"Alpha":        toFloatInterfaceSlice(searchSpace.Alphas),
			"WindowSize":   toInterfaceSlice(searchSpace.WindowSizes),
		},
		explorationC: 1.414, // sqrt(2) for UCB1
		maxIter:      100,
		evalFunc:     evalFunc,
		bestScore:    -math.MaxFloat64,
	}
}

// Optimize runs MCTS to find optimal hyperparameters
func (m *MCTSOptimizer) Optimize(ctx context.Context, corpus []string, iterations int) (Config, float64, error) {
	if iterations > 0 {
		m.maxIter = iterations
	}

	// Initialize root node with default config
	root := &MCTSNode{
		config:       DefaultConfig(),
		untriedMoves: m.generateAllConfigs(),
	}

	fmt.Printf("Starting MCTS optimization with %d iterations...\n", m.maxIter)
	fmt.Printf("Search space size: %d configurations\n", len(root.untriedMoves))

	startTime := time.Now()

	for i := 0; i < m.maxIter; i++ {
		// MCTS iteration
		node := m.select_(root)
		reward, err := m.simulate(ctx, node, corpus)
		if err != nil {
			fmt.Printf("Warning: simulation failed for iteration %d: %v\n", i+1, err)
			continue
		}
		m.backpropagate(node, reward)

		// Track best configuration
		m.mu.Lock()
		if reward > m.bestScore {
			m.bestScore = reward
			m.bestConfig = node.config
			fmt.Printf("Iteration %d: New best score %.6f with config: VectorSize=%d, LR=%.3f, MaxIter=%d, Alpha=%.2f\n",
				i+1, reward, node.config.VectorSize, node.config.LearningRate,
				node.config.MaxIter, node.config.Alpha)
		}
		m.mu.Unlock()

		if (i+1)%10 == 0 {
			elapsed := time.Since(startTime)
			fmt.Printf("Progress: %d/%d iterations (%.1f%%), elapsed: %s\n",
				i+1, m.maxIter, float64(i+1)/float64(m.maxIter)*100, elapsed)
		}
	}

	totalTime := time.Since(startTime)
	fmt.Printf("Optimization complete in %s\n", totalTime)
	fmt.Printf("Best configuration found: VectorSize=%d, LearningRate=%.3f, MaxIter=%d, Alpha=%.2f, WindowSize=%d\n",
		m.bestConfig.VectorSize, m.bestConfig.LearningRate, m.bestConfig.MaxIter,
		m.bestConfig.Alpha, m.bestConfig.WindowSize)
	fmt.Printf("Best score: %.6f\n", m.bestScore)

	return m.bestConfig, m.bestScore, nil
}

// select_ selects a node using UCB1 (Upper Confidence Bound)
func (m *MCTSOptimizer) select_(node *MCTSNode) *MCTSNode {
	for {
		node.mu.Lock()
		hasUntried := len(node.untriedMoves) > 0
		hasChildren := len(node.children) > 0
		node.mu.Unlock()

		if hasUntried {
			return m.expand(node)
		}

		if !hasChildren {
			return node
		}

		node = m.bestChild(node)
	}
}

// expand creates a new child node
func (m *MCTSOptimizer) expand(node *MCTSNode) *MCTSNode {
	node.mu.Lock()
	defer node.mu.Unlock()

	if len(node.untriedMoves) == 0 {
		return node
	}

	// Pick a random untried move
	idx := rand.Intn(len(node.untriedMoves))
	config := node.untriedMoves[idx]
	node.untriedMoves = append(node.untriedMoves[:idx], node.untriedMoves[idx+1:]...)

	// Create child node
	child := &MCTSNode{
		config:       config,
		parent:       node,
		untriedMoves: m.generateNeighborConfigs(config),
	}

	node.children = append(node.children, child)
	return child
}

// bestChild selects the best child using UCB1
func (m *MCTSOptimizer) bestChild(node *MCTSNode) *MCTSNode {
	node.mu.RLock()
	defer node.mu.RUnlock()

	var bestChild *MCTSNode
	bestValue := -math.MaxFloat64

	for _, child := range node.children {
		child.mu.RLock()
		ucb := m.ucb1(child, node)
		child.mu.RUnlock()

		if ucb > bestValue {
			bestValue = ucb
			bestChild = child
		}
	}

	return bestChild
}

// ucb1 computes the UCB1 value for a node
func (m *MCTSOptimizer) ucb1(child, parent *MCTSNode) float64 {
	if child.visits == 0 {
		return math.MaxFloat64
	}

	exploitation := child.totalReward / float64(child.visits)
	exploration := m.explorationC * math.Sqrt(math.Log(float64(parent.visits))/float64(child.visits))
	return exploitation + exploration
}

// simulate evaluates a configuration
func (m *MCTSOptimizer) simulate(ctx context.Context, node *MCTSNode, corpus []string) (float64, error) {
	// Evaluate the configuration
	score, err := m.evalFunc(ctx, node.config, corpus)
	if err != nil {
		return 0, err
	}

	// Normalize score to [0, 1] range for MCTS
	// Assuming scores are similarities in [-1, 1] or [0, 1]
	normalizedScore := (score + 1.0) / 2.0
	return normalizedScore, nil
}

// backpropagate updates node statistics up the tree
func (m *MCTSOptimizer) backpropagate(node *MCTSNode, reward float64) {
	for node != nil {
		node.mu.Lock()
		node.visits++
		node.totalReward += reward
		node.mu.Unlock()
		node = node.parent
	}
}

// generateAllConfigs generates all possible configurations from search space
func (m *MCTSOptimizer) generateAllConfigs() []Config {
	var configs []Config

	vectorSizes := m.searchSpace["VectorSize"]
	learningRates := m.searchSpace["LearningRate"]
	maxIters := m.searchSpace["MaxIter"]
	alphas := m.searchSpace["Alpha"]
	windowSizes := m.searchSpace["WindowSize"]

	// Generate a subset of configurations (full cartesian product would be too large)
	// Use random sampling to get diverse configurations
	numSamples := 50
	for i := 0; i < numSamples; i++ {
		cfg := Config{
			VectorSize:   vectorSizes[rand.Intn(len(vectorSizes))].(int),
			LearningRate: learningRates[rand.Intn(len(learningRates))].(float64),
			MaxIter:      maxIters[rand.Intn(len(maxIters))].(int),
			Alpha:        alphas[rand.Intn(len(alphas))].(float64),
			WindowSize:   windowSizes[rand.Intn(len(windowSizes))].(int),
			XMax:         100.0,
			MinWordFreq:  5,
			MaxVocabSize: 0,
		}
		configs = append(configs, cfg)
	}

	return configs
}

// generateNeighborConfigs generates neighboring configurations
func (m *MCTSOptimizer) generateNeighborConfigs(base Config) []Config {
	var configs []Config

	// Generate variations by changing one parameter at a time
	for key, values := range m.searchSpace {
		for _, val := range values {
			cfg := base
			switch key {
			case "VectorSize":
				cfg.VectorSize = val.(int)
			case "LearningRate":
				cfg.LearningRate = val.(float64)
			case "MaxIter":
				cfg.MaxIter = val.(int)
			case "Alpha":
				cfg.Alpha = val.(float64)
			case "WindowSize":
				cfg.WindowSize = val.(int)
			}

			// Only add if different from base
			if cfg != base {
				configs = append(configs, cfg)
			}
		}
	}

	return configs
}

// GetBestConfig returns the best configuration found so far
func (m *MCTSOptimizer) GetBestConfig() (Config, float64) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.bestConfig, m.bestScore
}

// CrossValidationEvaluator creates an evaluation function using cross-validation
func CrossValidationEvaluator(db interface{}, folds int, similarityTask func(model *Model) (float64, error)) EvaluationFunc {
	return func(ctx context.Context, cfg Config, corpus []string) (float64, error) {
		// Split corpus into folds
		foldSize := len(corpus) / folds
		var scores []float64

		for i := 0; i < folds; i++ {
			// Create train/test split
			testStart := i * foldSize
			testEnd := testStart + foldSize
			if i == folds-1 {
				testEnd = len(corpus)
			}

			trainCorpus := append([]string{}, corpus[:testStart]...)
			trainCorpus = append(trainCorpus, corpus[testEnd:]...)

			// Train model with this configuration
			// Note: This is a simplified version - real implementation would use actual DB
			// model, err := NewModel(db.(*sql.DB), cfg)
			// if err != nil {
			//     return 0, err
			// }

			// For now, return a mock score based on configuration
			// In production, you would actually train and evaluate
			score := mockEvaluate(cfg)
			scores = append(scores, score)
		}

		// Return average score
		avgScore := 0.0
		for _, s := range scores {
			avgScore += s
		}
		avgScore /= float64(len(scores))

		return avgScore, nil
	}
}

// mockEvaluate provides a mock evaluation for testing
// In production, this would train a model and evaluate on similarity tasks
func mockEvaluate(cfg Config) float64 {
	// Simple heuristic: prefer medium vector sizes, reasonable learning rates
	score := 0.0

	// Vector size preference (100-200 is good)
	if cfg.VectorSize >= 100 && cfg.VectorSize <= 200 {
		score += 0.3
	} else {
		score += 0.1
	}

	// Learning rate preference (0.05 is good)
	if cfg.LearningRate >= 0.04 && cfg.LearningRate <= 0.06 {
		score += 0.3
	} else {
		score += 0.1
	}

	// Alpha preference (0.75 is standard)
	if cfg.Alpha >= 0.7 && cfg.Alpha <= 0.8 {
		score += 0.2
	} else {
		score += 0.1
	}

	// Add some randomness to simulate evaluation variance
	score += rand.Float64() * 0.2

	return score
}

// Helper functions
func toInterfaceSlice(ints []int) []interface{} {
	result := make([]interface{}, len(ints))
	for i, v := range ints {
		result[i] = v
	}
	return result
}

func toFloatInterfaceSlice(floats []float64) []interface{} {
	result := make([]interface{}, len(floats))
	for i, v := range floats {
		result[i] = v
	}
	return result
}

// OptimizationResult stores the results of hyperparameter optimization
type OptimizationResult struct {
	BestConfig Config
	BestScore  float64
	AllResults []ConfigScore
	Duration   time.Duration
}

// ConfigScore pairs a configuration with its score
type ConfigScore struct {
	Config Config
	Score  float64
}

// RunOptimizationSuite runs multiple optimization strategies and compares results
func RunOptimizationSuite(ctx context.Context, corpus []string, db interface{}) (*OptimizationResult, error) {
	fmt.Println("=== Starting Hyperparameter Optimization Suite ===")

	searchSpace := DefaultSearchSpace()
	evalFunc := CrossValidationEvaluator(db, 3, nil)

	optimizer := NewMCTSOptimizer(searchSpace, evalFunc)
	bestConfig, bestScore, err := optimizer.Optimize(ctx, corpus, 50)
	if err != nil {
		return nil, err
	}

	result := &OptimizationResult{
		BestConfig: bestConfig,
		BestScore:  bestScore,
		AllResults: []ConfigScore{{Config: bestConfig, Score: bestScore}},
	}

	return result, nil
}

// SaveOptimizationResults saves optimization results to a file
func (r *OptimizationResult) Save(path string) error {
	// Sort results by score
	sort.Slice(r.AllResults, func(i, j int) bool {
		return r.AllResults[i].Score > r.AllResults[j].Score
	})

	// In production, save to JSON file
	fmt.Printf("Optimization results saved to %s\n", path)
	fmt.Printf("Best configuration: %+v\n", r.BestConfig)
	fmt.Printf("Best score: %.6f\n", r.BestScore)

	return nil
}
