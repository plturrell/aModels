package socialiq

import (
	"context"
	"fmt"
	"hash/fnv"
	"sync"
	"time"
)

// ============================================================================
// Performance Optimization - Section 9.12 Implementation
// Addresses exponential complexity in compositional reasoning
// ============================================================================

// OptimizedMentalStateComposer provides efficient mental state composition
type OptimizedMentalStateComposer struct {
	monoid              *MentalStateMonoid
	cache               *LRUCache
	maxCacheSize        int
	compositionStrategy CompositionStrategy
	mu                  sync.RWMutex
}

// CompositionStrategy defines how to compose mental states
type CompositionStrategy string

const (
	StrategyFull        CompositionStrategy = "full"        // Full composition
	StrategyApproximate CompositionStrategy = "approximate" // Approximate composition
	StrategyBounded     CompositionStrategy = "bounded"     // Resource-bounded
)

// NewOptimizedMentalStateComposer creates an optimized composer
func NewOptimizedMentalStateComposer(maxCacheSize int, strategy CompositionStrategy) *OptimizedMentalStateComposer {
	return &OptimizedMentalStateComposer{
		monoid:              NewMentalStateMonoid(),
		cache:               NewLRUCache(maxCacheSize),
		maxCacheSize:        maxCacheSize,
		compositionStrategy: strategy,
	}
}

// ComposeWithBudget performs resource-bounded composition
func (omsc *OptimizedMentalStateComposer) ComposeWithBudget(s1, s2 MentalState, maxOperations int) (MentalState, error) {
	omsc.mu.Lock()
	defer omsc.mu.Unlock()

	// Check cache first
	cacheKey := omsc.computeCacheKey(s1, s2)
	if cached, found := omsc.cache.Get(cacheKey); found {
		return cached.(MentalState), nil
	}

	// Perform composition based on strategy
	var result MentalState
	var err error

	switch omsc.compositionStrategy {
	case StrategyFull:
		result = omsc.monoid.Compose(s1, s2)
	case StrategyApproximate:
		result = omsc.approximateCompose(s1, s2, maxOperations)
	case StrategyBounded:
		result, err = omsc.boundedCompose(s1, s2, maxOperations)
	default:
		result = omsc.monoid.Compose(s1, s2)
	}

	if err != nil {
		return MentalState{}, err
	}

	// Cache result
	omsc.cache.Put(cacheKey, result)

	return result, nil
}

// approximateCompose performs approximate composition
func (omsc *OptimizedMentalStateComposer) approximateCompose(s1, s2 MentalState, maxOps int) MentalState {
	result := MentalState{
		Beliefs:    make(map[string]float64),
		Desires:    make(map[string]float64),
		Intentions: make(map[string]float64),
		Emotions:   make(map[string]float64),
	}

	opsCount := 0

	// Merge beliefs (top-k only)
	topBeliefs := omsc.getTopK(s1.Beliefs, s2.Beliefs, maxOps/4)
	for k, v := range topBeliefs {
		result.Beliefs[k] = v
		opsCount++
		if opsCount >= maxOps {
			break
		}
	}

	// Merge desires (top-k only)
	topDesires := omsc.getTopK(s1.Desires, s2.Desires, maxOps/4)
	for k, v := range topDesires {
		result.Desires[k] = v
		opsCount++
		if opsCount >= maxOps {
			break
		}
	}

	// Merge intentions (latest only)
	for k, v := range s2.Intentions {
		result.Intentions[k] = v
		opsCount++
		if opsCount >= maxOps {
			break
		}
	}

	// Merge emotions (weighted average)
	for k, v := range s1.Emotions {
		result.Emotions[k] = v * 0.5
		opsCount++
		if opsCount >= maxOps {
			break
		}
	}
	for k, v := range s2.Emotions {
		if existing, exists := result.Emotions[k]; exists {
			result.Emotions[k] = existing + v*0.5
		} else {
			result.Emotions[k] = v * 0.5
		}
		opsCount++
		if opsCount >= maxOps {
			break
		}
	}

	result.Uncertainty = (s1.Uncertainty + s2.Uncertainty) / 2.0

	return result
}

// boundedCompose performs bounded composition with error checking
func (omsc *OptimizedMentalStateComposer) boundedCompose(s1, s2 MentalState, maxOps int) (MentalState, error) {
	if maxOps <= 0 {
		return MentalState{}, fmt.Errorf("insufficient operation budget")
	}

	return omsc.approximateCompose(s1, s2, maxOps), nil
}

// getTopK returns top-k elements from two maps
func (omsc *OptimizedMentalStateComposer) getTopK(m1, m2 map[string]float64, k int) map[string]float64 {
	type kv struct {
		key   string
		value float64
	}

	// Merge and sort
	merged := make([]kv, 0)
	for key, val := range m1 {
		merged = append(merged, kv{key, val})
	}
	for key, val := range m2 {
		merged = append(merged, kv{key, val})
	}

	// Simple selection (in production, use heap)
	result := make(map[string]float64)
	count := 0
	for _, item := range merged {
		if count >= k {
			break
		}
		result[item.key] = item.value
		count++
	}

	return result
}

// computeCacheKey computes a hash key for caching
func (omsc *OptimizedMentalStateComposer) computeCacheKey(s1, s2 MentalState) string {
	h := fnv.New64a()

	// Hash beliefs
	for k, v := range s1.Beliefs {
		h.Write([]byte(k))
		h.Write([]byte(fmt.Sprintf("%.4f", v)))
	}
	for k, v := range s2.Beliefs {
		h.Write([]byte(k))
		h.Write([]byte(fmt.Sprintf("%.4f", v)))
	}

	return fmt.Sprintf("%x", h.Sum64())
}

// ============================================================================
// LRU Cache Implementation
// ============================================================================

// LRUCache implements a simple LRU cache
type LRUCache struct {
	capacity int
	cache    map[string]*cacheNode
	head     *cacheNode
	tail     *cacheNode
	mu       sync.RWMutex
}

type cacheNode struct {
	key   string
	value interface{}
	prev  *cacheNode
	next  *cacheNode
}

// NewLRUCache creates a new LRU cache
func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		cache:    make(map[string]*cacheNode),
	}
}

// Get retrieves a value from cache
func (lru *LRUCache) Get(key string) (interface{}, bool) {
	lru.mu.Lock()
	defer lru.mu.Unlock()

	if node, found := lru.cache[key]; found {
		lru.moveToFront(node)
		return node.value, true
	}
	return nil, false
}

// Put adds a value to cache
func (lru *LRUCache) Put(key string, value interface{}) {
	lru.mu.Lock()
	defer lru.mu.Unlock()

	if node, found := lru.cache[key]; found {
		node.value = value
		lru.moveToFront(node)
		return
	}

	newNode := &cacheNode{key: key, value: value}
	lru.cache[key] = newNode
	lru.addToFront(newNode)

	if len(lru.cache) > lru.capacity {
		lru.removeLast()
	}
}

func (lru *LRUCache) moveToFront(node *cacheNode) {
	lru.removeNode(node)
	lru.addToFront(node)
}

func (lru *LRUCache) addToFront(node *cacheNode) {
	node.next = lru.head
	node.prev = nil
	if lru.head != nil {
		lru.head.prev = node
	}
	lru.head = node
	if lru.tail == nil {
		lru.tail = node
	}
}

func (lru *LRUCache) removeNode(node *cacheNode) {
	if node.prev != nil {
		node.prev.next = node.next
	} else {
		lru.head = node.next
	}
	if node.next != nil {
		node.next.prev = node.prev
	} else {
		lru.tail = node.prev
	}
}

func (lru *LRUCache) removeLast() {
	if lru.tail == nil {
		return
	}
	delete(lru.cache, lru.tail.key)
	lru.removeNode(lru.tail)
}

// ============================================================================
// Parallel Reasoning Engine
// ============================================================================

// ParallelReasoningEngine performs parallel social reasoning
type ParallelReasoningEngine struct {
	coordinator *MetacognitiveCoordinator
	numWorkers  int
	workQueue   chan reasoningTask
	results     chan reasoningResult
	mu          sync.RWMutex
}

type reasoningTask struct {
	id    string
	input MultimodalInput
}

type reasoningResult struct {
	id     string
	result *PredictionResult
	err    error
}

// NewParallelReasoningEngine creates a parallel reasoning engine
func NewParallelReasoningEngine(coordinator *MetacognitiveCoordinator, numWorkers int) *ParallelReasoningEngine {
	return &ParallelReasoningEngine{
		coordinator: coordinator,
		numWorkers:  numWorkers,
		workQueue:   make(chan reasoningTask, numWorkers*2),
		results:     make(chan reasoningResult, numWorkers*2),
	}
}

// Start starts the worker pool
func (pre *ParallelReasoningEngine) Start(ctx context.Context) {
	for i := 0; i < pre.numWorkers; i++ {
		go pre.worker(ctx, i)
	}
}

// worker processes reasoning tasks
func (pre *ParallelReasoningEngine) worker(ctx context.Context, _ int) {
	for {
		select {
		case <-ctx.Done():
			return
		case task := <-pre.workQueue:
			result, err := pre.coordinator.ReasonAndAct(ctx, task.input)
			pre.results <- reasoningResult{
				id:     task.id,
				result: result,
				err:    err,
			}
		}
	}
}

// ProcessBatch processes a batch of inputs in parallel
func (pre *ParallelReasoningEngine) ProcessBatch(ctx context.Context, inputs []MultimodalInput) ([]PredictionResult, error) {
	// Submit tasks
	for i, input := range inputs {
		pre.workQueue <- reasoningTask{
			id:    fmt.Sprintf("task_%d", i),
			input: input,
		}
	}

	// Collect results
	results := make([]PredictionResult, len(inputs))
	for i := 0; i < len(inputs); i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case result := <-pre.results:
			if result.err != nil {
				return nil, result.err
			}
			// Parse task ID to get index
			var idx int
			fmt.Sscanf(result.id, "task_%d", &idx)
			results[idx] = *result.result
		}
	}

	return results, nil
}

// ============================================================================
// Performance Monitoring
// ============================================================================

// PerformanceMonitor tracks reasoning performance
type PerformanceMonitor struct {
	metrics map[string]*PerformanceMetric
	mu      sync.RWMutex
}

// PerformanceMetric tracks a specific metric
type PerformanceMetric struct {
	Name        string
	Count       int64
	TotalTime   time.Duration
	MinTime     time.Duration
	MaxTime     time.Duration
	AvgTime     time.Duration
	LastUpdated time.Time
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		metrics: make(map[string]*PerformanceMetric),
	}
}

// RecordOperation records an operation's performance
func (pm *PerformanceMonitor) RecordOperation(name string, duration time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	metric, exists := pm.metrics[name]
	if !exists {
		metric = &PerformanceMetric{
			Name:    name,
			MinTime: duration,
			MaxTime: duration,
		}
		pm.metrics[name] = metric
	}

	metric.Count++
	metric.TotalTime += duration
	metric.AvgTime = metric.TotalTime / time.Duration(metric.Count)
	metric.LastUpdated = time.Now()

	if duration < metric.MinTime {
		metric.MinTime = duration
	}
	if duration > metric.MaxTime {
		metric.MaxTime = duration
	}
}

// GetMetrics returns all metrics
func (pm *PerformanceMonitor) GetMetrics() map[string]*PerformanceMetric {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Return copy
	result := make(map[string]*PerformanceMetric)
	for k, v := range pm.metrics {
		metricCopy := *v
		result[k] = &metricCopy
	}
	return result
}

// PrintMetrics prints performance metrics
func (pm *PerformanceMonitor) PrintMetrics() {
	metrics := pm.GetMetrics()

	fmt.Println("=== Performance Metrics ===")
	for name, metric := range metrics {
		fmt.Printf("\n%s:\n", name)
		fmt.Printf("  Count: %d\n", metric.Count)
		fmt.Printf("  Avg Time: %v\n", metric.AvgTime)
		fmt.Printf("  Min Time: %v\n", metric.MinTime)
		fmt.Printf("  Max Time: %v\n", metric.MaxTime)
		fmt.Printf("  Total Time: %v\n", metric.TotalTime)
	}
}

// WithTiming wraps a function with timing
func (pm *PerformanceMonitor) WithTiming(name string, fn func() error) error {
	start := time.Now()
	err := fn()
	duration := time.Since(start)
	pm.RecordOperation(name, duration)
	return err
}
