package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

func main() {
	fmt.Println("Maths Package - Batch Processing Example")
	fmt.Println(strings.Repeat("=", 50))

	// Create test data
	dimension := 128
	numVectors := 1000
	numQueries := 100

	vectors := createRandomVectors(numVectors, dimension)
	queries := createRandomVectors(numQueries, dimension)

	fmt.Printf("Processing %d queries against %d vectors (dimension: %d)\n",
		numQueries, numVectors, dimension)

	// Test 1: Individual processing
	fmt.Println("\n1. Individual Processing:")
	start := time.Now()
	individualResults := processIndividual(queries, vectors)
	individualTime := time.Since(start)
	fmt.Printf("Individual processing time: %v\n", individualTime)
	fmt.Printf("Total similarities computed: %d\n", len(individualResults))

	// Test 2: Batch processing
	fmt.Println("\n2. Batch Processing:")
	start = time.Now()
	batchResults := processBatch(queries, vectors)
	batchTime := time.Since(start)
	fmt.Printf("Batch processing time: %v\n", batchTime)
	fmt.Printf("Total similarities computed: %d\n", len(batchResults))

	// Performance comparison
	fmt.Println("\n3. Performance Comparison:")
	speedup := float64(individualTime) / float64(batchTime)
	fmt.Printf("Batch processing speedup: %.2fx\n", speedup)

	// Verify results are consistent
	fmt.Println("\n4. Result Verification:")
	verifyResults(individualResults, batchResults)

	// Test 3: Memory efficiency
	fmt.Println("\n5. Memory Efficiency:")
	testMemoryEfficiency(queries, vectors)

	fmt.Println("\nBatch processing example completed!")
}

func createRandomVectors(count, dimension int) [][]float64 {
	rand.Seed(time.Now().UnixNano())
	vectors := make([][]float64, count)
	for i := 0; i < count; i++ {
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rand.Float64()*2 - 1 // Random values between -1 and 1
		}
	}
	return vectors
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

func processIndividual(queries, vectors [][]float64) []float64 {
	var results []float64

	for _, query := range queries {
		for _, vector := range vectors {
			similarity := maths.CosineAuto(query, vector)
			results = append(results, similarity)
		}
	}

	return results
}

func processBatch(queries, vectors [][]float64) []float64 {
	var results []float64
	flatVectors := flattenVectors(vectors)
	numVectors := len(vectors)

	for _, query := range queries {
		repeated := repeatVector(query, numVectors)
		similarities := maths.CosineBatchAuto(len(query), flatVectors, repeated)
		results = append(results, similarities...)
	}

	return results
}

func verifyResults(individual, batch []float64) {
	if len(individual) != len(batch) {
		fmt.Printf("❌ Length mismatch: individual=%d, batch=%d\n", len(individual), len(batch))
		return
	}

	maxDiff := 0.0
	for i := 0; i < len(individual); i++ {
		diff := abs(individual[i] - batch[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	fmt.Printf("Maximum difference: %.10f\n", maxDiff)
	if maxDiff < 1e-10 {
		fmt.Println("✅ Results are consistent")
	} else {
		fmt.Println("❌ Results differ significantly")
	}
}

func testMemoryEfficiency(queries, vectors [][]float64) {
	// Test with different batch sizes
	batchSizes := []int{10, 50, 100, 200}
	flatVectors := flattenVectors(vectors)
	numVectors := len(vectors)

	for _, batchSize := range batchSizes {
		start := time.Now()

		// Process queries in batches
		for i := 0; i < len(queries); i += batchSize {
			end := i + batchSize
			if end > len(queries) {
				end = len(queries)
			}

			batch := queries[i:end]
			for _, query := range batch {
				repeated := repeatVector(query, numVectors)
				_ = maths.CosineBatchAuto(len(query), flatVectors, repeated)
			}
		}

		duration := time.Since(start)
		fmt.Printf("Batch size %d: %v\n", batchSize, duration)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
