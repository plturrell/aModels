package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

type SearchResult struct {
	Index int
	Score float64
}

func main() {
	fmt.Println("Maths Package - Vector Search Example")
	fmt.Println(strings.Repeat("=", 50))

	// Create sample vector database
	rand.Seed(time.Now().UnixNano())
	vectorDB := createVectorDatabase(1000, 128) // 1000 vectors of dimension 128

	// Create query vector
	query := createRandomVector(128)

	fmt.Printf("Vector database: %d vectors of dimension %d\n", len(vectorDB), len(vectorDB[0]))
	fmt.Printf("Query vector dimension: %d\n", len(query))

	// Test 1: Individual similarity search
	fmt.Println("\n1. Individual Similarity Search:")
	start := time.Now()
	results := findSimilarVectors(query, vectorDB, 0.8, 10)
	duration := time.Since(start)

	fmt.Printf("Found %d similar vectors (threshold: 0.8)\n", len(results))
	fmt.Printf("Search time: %v\n", duration)

	// Show top 5 results
	fmt.Println("Top 5 results:")
	for i, result := range results[:min(5, len(results))] {
		fmt.Printf("  %d. Index: %d, Score: %.4f\n", i+1, result.Index, result.Score)
	}

	// Test 2: Batch similarity search
	fmt.Println("\n2. Batch Similarity Search:")
	start = time.Now()
	batchResults := findSimilarVectorsBatch(query, vectorDB, 0.8, 10)
	batchDuration := time.Since(start)

	fmt.Printf("Found %d similar vectors (batch)\n", len(batchResults))
	fmt.Printf("Batch search time: %v\n", batchDuration)
	fmt.Printf("Speedup: %.2fx\n", float64(duration)/float64(batchDuration))

	// Test 3: Quantized search
	fmt.Println("\n3. Quantized Search:")
	quantizedResults := findSimilarVectorsQuantized(query, vectorDB, 10)
	fmt.Printf("Found %d similar vectors (quantized)\n", len(quantizedResults))

	// Show top 5 quantized results
	fmt.Println("Top 5 quantized results:")
	for i, result := range quantizedResults[:min(5, len(quantizedResults))] {
		fmt.Printf("  %d. Index: %d, Score: %.4f\n", i+1, result.Index, result.Score)
	}

	// Test 4: Performance comparison
	fmt.Println("\n4. Performance Comparison:")
	compareSearchMethods(query, vectorDB)

	// Test 5: Cache performance
	fmt.Println("\n5. Cache Performance:")
	testCachePerformance(query, vectorDB)

	fmt.Println("\nVector search example completed!")
}

func createVectorDatabase(numVectors, dimension int) [][]float64 {
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = createRandomVector(dimension)
	}
	return vectors
}

func createRandomVector(dimension int) []float64 {
	vector := make([]float64, dimension)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1 // Random values between -1 and 1
	}
	return vector
}

func flattenVectors(vectors [][]float64) []float64 {
	if len(vectors) == 0 {
		return nil
	}
	flat := make([]float64, 0, len(vectors)*len(vectors[0]))
	for _, vec := range vectors {
		flat = append(flat, vec...)
	}
	return flat
}

func repeatVector(vec []float64, count int) []float64 {
	repeated := make([]float64, 0, len(vec)*count)
	for i := 0; i < count; i++ {
		repeated = append(repeated, vec...)
	}
	return repeated
}

func flattenInt8Matrix(matrix [][]int8) []int8 {
	if len(matrix) == 0 {
		return nil
	}
	flat := make([]int8, 0, len(matrix)*len(matrix[0]))
	for _, row := range matrix {
		flat = append(flat, row...)
	}
	return flat
}

func findSimilarVectors(query []float64, candidates [][]float64, threshold float64, topK int) []SearchResult {
	var results []SearchResult

	for i, candidate := range candidates {
		similarity := maths.CosineAuto(query, candidate)
		if similarity >= threshold {
			results = append(results, SearchResult{
				Index: i,
				Score: similarity,
			})
		}
	}

	// Sort by score (descending)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score < results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if len(results) > topK {
		return results[:topK]
	}
	return results
}

func findSimilarVectorsBatch(query []float64, candidates [][]float64, threshold float64, topK int) []SearchResult {
	flatCandidates := flattenVectors(candidates)
	repeatedQuery := repeatVector(query, len(candidates))
	similarities := maths.CosineBatchAuto(len(query), flatCandidates, repeatedQuery)

	var results []SearchResult
	for i, similarity := range similarities {
		if similarity >= threshold {
			results = append(results, SearchResult{
				Index: i,
				Score: similarity,
			})
		}
	}

	// Sort by score (descending)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score < results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if len(results) > topK {
		return results[:topK]
	}
	return results
}

func findSimilarVectorsQuantized(query []float64, candidates [][]float64, topK int) []SearchResult {
	// Create tensor operations instance
	ops := maths.NewTensorOps(false)

	// Quantize vectors
	quantized, _, err := ops.QuantizeInt8(candidates)
	if err != nil {
		log.Printf("Quantization failed: %v", err)
		return []SearchResult{}
	}

	// Use quantized search
	flatQuantized := flattenInt8Matrix(quantized)
	indices, scores := maths.CosineTopKInt8(len(query), flatQuantized, query, topK)

	var results []SearchResult
	for i, idx := range indices {
		results = append(results, SearchResult{
			Index: idx,
			Score: scores[i],
		})
	}

	return results
}

func compareSearchMethods(query []float64, candidates [][]float64) {
	iterations := 100

	// Individual search
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = findSimilarVectors(query, candidates, 0.8, 10)
	}
	individualTime := time.Since(start)

	// Batch search
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_ = findSimilarVectorsBatch(query, candidates, 0.8, 10)
	}
	batchTime := time.Since(start)

	fmt.Printf("Individual search (%d iterations): %v\n", iterations, individualTime)
	fmt.Printf("Batch search (%d iterations): %v\n", iterations, batchTime)
	fmt.Printf("Batch speedup: %.2fx\n", float64(individualTime)/float64(batchTime))
}

func testCachePerformance(query []float64, candidates [][]float64) {
	// Clear cache
	maths.ClearCache()

	// First run (cache miss)
	start := time.Now()
	_ = findSimilarVectors(query, candidates[:100], 0.8, 10)
	firstRun := time.Since(start)

	// Second run (cache hit)
	start = time.Now()
	_ = findSimilarVectors(query, candidates[:100], 0.8, 10)
	secondRun := time.Since(start)

	fmt.Printf("First run (cache miss): %v\n", firstRun)
	fmt.Printf("Second run (cache hit): %v\n", secondRun)
	fmt.Printf("Cache speedup: %.2fx\n", float64(firstRun)/float64(secondRun))

	// Show cache stats
	stats := maths.GetCacheStats()
	fmt.Printf("Cache stats: %+v\n", stats)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
