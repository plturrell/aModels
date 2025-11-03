package piqa

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
)

// ANNRetriever implements Approximate Nearest Neighbor search for scalable retrieval
type ANNRetriever struct {
	storage      EmbeddingStorage
	index        *LSHIndex
	contextCache map[string]*ContextEmbeddings
	mu           sync.RWMutex
	cacheEnabled bool
	cacheTTL     int64 // TTL in seconds
	cacheAccess  map[string]int64
}

// LSHIndex implements Locality-Sensitive Hashing for ANN search
type LSHIndex struct {
	numTables     int
	numHashes     int
	dimension     int
	tables        []map[uint64][]string
	hashFunctions [][]*LSHHashFunc
	mu            sync.RWMutex
}

// LSHHashFunc represents a single hash function for LSH
type LSHHashFunc struct {
	randomVector []float32
	bias         float32
	width        float32
}

// NewANNRetriever creates a new ANN-based retriever with caching
func NewANNRetriever(storage EmbeddingStorage, dimension int) *ANNRetriever {
	return &ANNRetriever{
		storage:      storage,
		index:        NewLSHIndex(10, 5, dimension), // 10 tables, 5 hashes per table
		contextCache: make(map[string]*ContextEmbeddings),
		cacheEnabled: true,
		cacheTTL:     3600, // 1 hour default
		cacheAccess:  make(map[string]int64),
	}
}

// NewLSHIndex creates a new LSH index
func NewLSHIndex(numTables, numHashes, dimension int) *LSHIndex {
	index := &LSHIndex{
		numTables:     numTables,
		numHashes:     numHashes,
		dimension:     dimension,
		tables:        make([]map[uint64][]string, numTables),
		hashFunctions: make([][]*LSHHashFunc, numTables),
	}

	// Initialize hash tables and functions
	for i := 0; i < numTables; i++ {
		index.tables[i] = make(map[uint64][]string)
		index.hashFunctions[i] = make([]*LSHHashFunc, numHashes)

		for j := 0; j < numHashes; j++ {
			index.hashFunctions[i][j] = newLSHHashFunc(dimension)
		}
	}

	return index
}

// newLSHHashFunc creates a new random hash function
func newLSHHashFunc(dimension int) *LSHHashFunc {
	vec := make([]float32, dimension)
	for i := range vec {
		vec[i] = float32(randNormal())
	}
	return &LSHHashFunc{
		randomVector: vec,
		bias:         float32(randUniform() * 4.0), // width = 4.0
		width:        4.0,
	}
}

// hash computes hash value for a vector
func (h *LSHHashFunc) hash(vec []float32) uint32 {
	dotProd := float32(0.0)
	for i := range vec {
		dotProd += vec[i] * h.randomVector[i]
	}
	return uint32(math.Floor(float64((dotProd + h.bias) / h.width)))
}

// IndexContext indexes all phrase embeddings from a context in the LSH structure
func (r *ANNRetriever) IndexContext(ctx context.Context, emb *ContextEmbeddings) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Index each phrase embedding
	for i := range emb.Phrases {
		if i >= len(emb.Embeddings) {
			continue
		}

		phrase := emb.Phrases[i]
		rawVector := emb.Embeddings[i]

		normalizedVec := make([]float32, len(rawVector))
		copy(normalizedVec, rawVector)
		normalizedVec = normalize(normalizedVec)

		// Create unique context ID for this phrase
		contextID := fmt.Sprintf("%s_%d_%d", emb.ParagraphID, phrase.Start, phrase.End)

		// Store in cache (create single-phrase ContextEmbeddings for retrieval)
		r.contextCache[contextID] = &ContextEmbeddings{
			ParagraphID: emb.ParagraphID,
			Phrases:     []Phrase{phrase},
			Embeddings:  [][]float32{normalizedVec},
		}

		// Index in LSH
		r.index.mu.Lock()
		for tableIdx := 0; tableIdx < r.index.numTables; tableIdx++ {
			hashValue := r.computeTableHash(normalizedVec, tableIdx)
			r.index.tables[tableIdx][hashValue] = append(
				r.index.tables[tableIdx][hashValue], contextID)
		}
		r.index.mu.Unlock()
	}

	return nil
}

// computeTableHash computes combined hash for a table
func (r *ANNRetriever) computeTableHash(vec []float32, tableIdx int) uint64 {
	var combined uint64
	for hashIdx, hashFunc := range r.index.hashFunctions[tableIdx] {
		h := hashFunc.hash(vec)
		combined = combined*31 + uint64(h)
		_ = hashIdx // Use if needed for debugging
	}
	return combined
}

// RetrieveTopK retrieves top-k most similar contexts using ANN
func (r *ANNRetriever) RetrieveTopK(ctx context.Context, queryVec []float32, k int) ([]*ContextEmbeddings, error) {
	normalizedQuery := make([]float32, len(queryVec))
	copy(normalizedQuery, queryVec)
	normalizedQuery = normalize(normalizedQuery)

	r.mu.RLock()
	defer r.mu.RUnlock()

	// Get candidate set from LSH
	candidates := r.getCandidates(normalizedQuery)

	if k <= 0 {
		k = 1
	}

	if len(candidates) < k {
		expanded := make([]string, 0, len(r.contextCache))
		seen := make(map[string]struct{}, len(r.contextCache))

		for _, id := range candidates {
			if _, exists := seen[id]; exists {
				continue
			}
			seen[id] = struct{}{}
			expanded = append(expanded, id)
		}

		for contextID := range r.contextCache {
			if _, exists := seen[contextID]; exists {
				continue
			}
			seen[contextID] = struct{}{}
			expanded = append(expanded, contextID)
		}

		candidates = expanded
	}

	// Compute exact similarities for candidates
	type scoredContext struct {
		context *ContextEmbeddings
		score   float64
	}

	scored := make([]scoredContext, 0, len(candidates))
	for _, contextID := range candidates {
		emb, ok := r.contextCache[contextID]
		if !ok {
			continue
		}

		// Get the first embedding (single-phrase context from cache)
		if len(emb.Embeddings) == 0 {
			continue
		}
		similarity := float64(cosineSimilarity(normalizedQuery, emb.Embeddings[0]))
		scored = append(scored, scoredContext{emb, similarity})
	}

	// Sort by score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Return top-k
	if k > len(scored) {
		k = len(scored)
	}

	results := make([]*ContextEmbeddings, k)
	for i := 0; i < k; i++ {
		results[i] = scored[i].context
	}

	return results, nil
}

// getCandidates retrieves candidate contexts from LSH tables
func (r *ANNRetriever) getCandidates(queryVec []float32) []string {
	r.index.mu.RLock()
	defer r.index.mu.RUnlock()

	candidateSet := make(map[string]bool)

	// Query each table
	for tableIdx := 0; tableIdx < r.index.numTables; tableIdx++ {
		hashValue := r.computeTableHash(queryVec, tableIdx)
		if contexts, ok := r.index.tables[tableIdx][hashValue]; ok {
			for _, contextID := range contexts {
				candidateSet[contextID] = true
			}
		}
	}

	// Convert to slice
	candidates := make([]string, 0, len(candidateSet))
	for contextID := range candidateSet {
		candidates = append(candidates, contextID)
	}

	return candidates
}

// cosineSimilarity is defined in encoder.go

// EvictCache evicts old entries from cache based on TTL
func (r *ANNRetriever) EvictCache(currentTime int64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for contextID, accessTime := range r.cacheAccess {
		if currentTime-accessTime > r.cacheTTL {
			delete(r.contextCache, contextID)
			delete(r.cacheAccess, contextID)
		}
	}
}

// ClearCache clears the entire cache
func (r *ANNRetriever) ClearCache() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.contextCache = make(map[string]*ContextEmbeddings)
	r.cacheAccess = make(map[string]int64)
}

// Helper functions for random number generation

func randNormal() float64 {
	// Box-Muller transform for normal distribution
	u1 := randUniform()
	u2 := randUniform()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func randUniform() float64 {
	// Simple uniform random in [0,1)
	// In production, use crypto/rand or math/rand with proper seeding
	return float64(uint32(0x12345678)%1000000) / 1000000.0
}
