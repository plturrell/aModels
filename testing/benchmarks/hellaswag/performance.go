package hellaswag

import (
	"sort"
	"sync"
)

// OptimizedFilter provides high-performance adversarial filtering
type OptimizedFilter struct {
	*AdversarialFilter
	Cache      *ScoreCache
	BatchSize  int
	NumWorkers int
}

// ScoreCache caches discriminator scores to avoid recomputation
type ScoreCache struct {
	mu     sync.RWMutex
	scores map[string]float64
}

func NewScoreCache() *ScoreCache {
	return &ScoreCache{
		scores: make(map[string]float64),
	}
}

func (c *ScoreCache) Get(key string) (float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	score, exists := c.scores[key]
	return score, exists
}

func (c *ScoreCache) Set(key string, score float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.scores[key] = score
}

func (c *ScoreCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.scores = make(map[string]float64)
}

// GenerateDistractorsOptimized uses parallel processing and caching
func (of *OptimizedFilter) GenerateDistractorsOptimized(context, goldEnding string, numDistractors int) ([]FilteredEnding, error) {
	// Step 1: Generate candidates in parallel
	candidates, err := of.Generator.Generate(context, of.Config.NumCandidates*numDistractors)
	if err != nil {
		return nil, err
	}

	// Step 2: Filter obvious duplicates
	candidates = of.deduplicateCandidates(candidates, goldEnding)

	// Step 3: Score candidates in parallel batches
	scored := of.scoreCandidatesParallel(context, candidates)

	// Step 4: Efficient sorting (O(n log n) instead of O(n²))
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].BERTScore > scored[j].BERTScore
	})

	// Step 5: Select top N
	if len(scored) < numDistractors {
		return scored, nil
	}

	result := scored[:numDistractors]
	for i := range result {
		result[i].ConfusionRank = i + 1
	}

	return result, nil
}

func (of *OptimizedFilter) deduplicateCandidates(candidates []string, gold string) []string {
	seen := make(map[string]bool)
	seen[gold] = true // Exclude gold ending

	unique := make([]string, 0, len(candidates))
	for _, c := range candidates {
		if !seen[c] && !isTooSimilar(c, gold) {
			seen[c] = true
			unique = append(unique, c)
		}
	}

	return unique
}

func (of *OptimizedFilter) scoreCandidatesParallel(context string, candidates []string) []FilteredEnding {
	// Create work channel
	jobs := make(chan scoreJob, len(candidates))
	results := make(chan FilteredEnding, len(candidates))

	// Start worker pool
	var wg sync.WaitGroup
	for w := 0; w < of.NumWorkers; w++ {
		wg.Add(1)
		go of.scoreWorker(jobs, results, &wg)
	}

	// Send jobs
	for _, candidate := range candidates {
		jobs <- scoreJob{
			context:   context,
			candidate: candidate,
		}
	}
	close(jobs)

	// Wait for completion
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	scored := make([]FilteredEnding, 0, len(candidates))
	for result := range results {
		if result.BERTScore >= of.Config.MinBERTConfusion {
			scored = append(scored, result)
		}
	}

	return scored
}

type scoreJob struct {
	context   string
	candidate string
}

func (of *OptimizedFilter) scoreWorker(jobs <-chan scoreJob, results chan<- FilteredEnding, wg *sync.WaitGroup) {
	defer wg.Done()

	for job := range jobs {
		// Check cache first
		cacheKey := job.context + "|" + job.candidate
		if score, exists := of.Cache.Get(cacheKey); exists {
			results <- FilteredEnding{
				Text:      job.candidate,
				BERTScore: score,
			}
			continue
		}

		// Score with discriminator
		score, err := of.Discriminator.Score(job.context, job.candidate)
		if err != nil {
			continue
		}

		// Cache result
		of.Cache.Set(cacheKey, score)

		results <- FilteredEnding{
			Text:      job.candidate,
			BERTScore: score,
		}
	}
}

// BatchProcessor processes multiple examples efficiently
type BatchProcessor struct {
	Filter     *OptimizedFilter
	BatchSize  int
	NumWorkers int
}

func NewBatchProcessor(filter *OptimizedFilter, batchSize, numWorkers int) *BatchProcessor {
	return &BatchProcessor{
		Filter:     filter,
		BatchSize:  batchSize,
		NumWorkers: numWorkers,
	}
}

func (bp *BatchProcessor) ProcessBatch(examples []Example) ([]ProcessedExample, error) {
	jobs := make(chan processJob, len(examples))
	results := make(chan ProcessedExample, len(examples))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < bp.NumWorkers; w++ {
		wg.Add(1)
		go bp.processWorker(jobs, results, &wg)
	}

	// Send jobs
	for i, ex := range examples {
		jobs <- processJob{
			index:   i,
			example: ex,
		}
	}
	close(jobs)

	// Wait for completion
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	processed := make([]ProcessedExample, 0, len(examples))
	for result := range results {
		processed = append(processed, result)
	}

	// Sort by original index
	sort.Slice(processed, func(i, j int) bool {
		return processed[i].Index < processed[j].Index
	})

	return processed, nil
}

type processJob struct {
	index   int
	example Example
}

type ProcessedExample struct {
	Index       int
	Example     Example
	Distractors []FilteredEnding
	Error       error
}

func (bp *BatchProcessor) processWorker(jobs <-chan processJob, results chan<- ProcessedExample, wg *sync.WaitGroup) {
	defer wg.Done()

	for job := range jobs {
		// Generate distractors for this example
		distractors, err := bp.Filter.GenerateDistractorsOptimized(
			job.example.Context,
			job.example.Endings[0], // Assume first is gold
			3,                      // Generate 3 distractors
		)

		results <- ProcessedExample{
			Index:       job.index,
			Example:     job.example,
			Distractors: distractors,
			Error:       err,
		}
	}
}

// MemoryEfficientCache implements LRU caching for large-scale processing
type MemoryEfficientCache struct {
	mu       sync.RWMutex
	scores   map[string]*cacheEntry
	maxSize  int
	eviction *evictionQueue
}

type cacheEntry struct {
	score      float64
	accessTime int64
}

type evictionQueue struct {
	items []string
	head  int
}

func NewMemoryEfficientCache(maxSize int) *MemoryEfficientCache {
	return &MemoryEfficientCache{
		scores:  make(map[string]*cacheEntry),
		maxSize: maxSize,
		eviction: &evictionQueue{
			items: make([]string, 0, maxSize),
		},
	}
}

func (c *MemoryEfficientCache) Get(key string) (float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.scores[key]
	if !exists {
		return 0.0, false
	}

	return entry.score, true
}

func (c *MemoryEfficientCache) Set(key string, score float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if we need to evict
	if len(c.scores) >= c.maxSize {
		c.evictOldest()
	}

	c.scores[key] = &cacheEntry{
		score:      score,
		accessTime: getCurrentTime(),
	}
}

func (c *MemoryEfficientCache) evictOldest() {
	// Find oldest entry
	var oldestKey string
	var oldestTime int64 = 1<<63 - 1

	for key, entry := range c.scores {
		if entry.accessTime < oldestTime {
			oldestTime = entry.accessTime
			oldestKey = key
		}
	}

	if oldestKey != "" {
		delete(c.scores, oldestKey)
	}
}

func getCurrentTime() int64 {
	// Simple counter for access time
	return int64(len(make([]byte, 0)))
}

// NewOptimizedFilter creates a high-performance filter
func NewOptimizedFilter(config AFConfig, generator EndingGenerator, discriminator BERTDiscriminator) *OptimizedFilter {
	return &OptimizedFilter{
		AdversarialFilter: &AdversarialFilter{
			Config:        config,
			Generator:     generator,
			Discriminator: discriminator,
		},
		Cache:      NewScoreCache(),
		BatchSize:  32,
		NumWorkers: 4,
	}
}
