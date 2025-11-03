package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

func main() {
	fmt.Println("Maths Package - Performance Monitoring Example")
	fmt.Println(strings.Repeat("=", 60))

	// Initialize monitoring
	fmt.Println("\n1. Initializing Performance Monitoring:")
	maths.SetCacheSize(1000)
	fmt.Println("Cache size set to 1000")

	// Show initial state
	fmt.Println("\n2. Initial State:")
	showMetrics()

	// Perform various operations to generate metrics
	fmt.Println("\n3. Performing Operations:")
	performOperations()

	// Show metrics after operations
	fmt.Println("\n4. Metrics After Operations:")
	showMetrics()

	// Test cache performance
	fmt.Println("\n5. Cache Performance Test:")
	testCachePerformance()

	// Test bottlenecks
	fmt.Println("\n6. Bottleneck Analysis:")
	analyzeBottlenecks()

	// Show operation heatmap
	fmt.Println("\n7. Operation Heatmap:")
	showHeatmap()

	fmt.Println("\nPerformance monitoring example completed!")
}

func showMetrics() {
	metrics := maths.GetPerformanceMetrics()
	fmt.Printf("Performance metrics: %+v\n", metrics)

	cacheStats := maths.GetCacheStats()
	fmt.Printf("Cache statistics: %+v\n", cacheStats)
}

func performOperations() {
	// Create test vectors
	vectors := createTestVectors(100, 64)
	query := createTestVector(64)
	flatVectors := flattenVectors(vectors)
	repeatedQuery := repeatVector(query, len(vectors))

	// Perform various operations
	fmt.Println("  - Computing cosine similarities...")
	for i := 0; i < 50; i++ {
		_ = maths.CosineAuto(query, vectors[i%len(vectors)])
	}

	fmt.Println("  - Computing dot products...")
	for i := 0; i < 50; i++ {
		_ = maths.DotAuto(query, vectors[i%len(vectors)])
	}

	fmt.Println("  - Computing batch similarities...")
	for i := 0; i < 10; i++ {
		_ = maths.CosineBatchAuto(len(query), flatVectors, repeatedQuery)
	}

	fmt.Println("  - Performing statistical operations...")
	values := make([]float64, 1000)
	for i := range values {
		values[i] = rand.Float64()
	}
	_ = maths.Sum(values)
	_ = maths.Mean(values)
	_ = maths.Min(values)
	_ = maths.Max(values)
}

func testCachePerformance() {
	// Clear cache first
	maths.ClearCache()

	vectors := createTestVectors(100, 64)
	query := createTestVector(64)

	// First run (cache miss)
	start := time.Now()
	for i := 0; i < 20; i++ {
		_ = maths.CosineAuto(query, vectors[i%len(vectors)])
	}
	firstRun := time.Since(start)

	// Second run (cache hit)
	start = time.Now()
	for i := 0; i < 20; i++ {
		_ = maths.CosineAuto(query, vectors[i%len(vectors)])
	}
	secondRun := time.Since(start)

	fmt.Printf("First run (cache miss): %v\n", firstRun)
	fmt.Printf("Second run (cache hit): %v\n", secondRun)
	if secondRun > 0 {
		speedup := float64(firstRun) / float64(secondRun)
		fmt.Printf("Cache speedup: %.2fx\n", speedup)
	}
}

func analyzeBottlenecks() {
	bottlenecks := maths.GetBottlenecks()
	if len(bottlenecks) == 0 {
		fmt.Println("No bottlenecks detected")
	} else {
		fmt.Printf("Detected bottlenecks: %v\n", bottlenecks)
	}

	// Show top operations
	topOps := maths.GetTopOperations(5)
	fmt.Println("Top 5 operations:")
	for i, op := range topOps {
		fmt.Printf("  %d. %+v\n", i+1, op)
	}
}

func showHeatmap() {
	heatmap := maths.GetOperationHeatmap()
	if len(heatmap) == 0 {
		fmt.Println("No heatmap data available")
	} else {
		fmt.Printf("Operation heatmap: %+v\n", heatmap)
	}
}

func createTestVectors(count, dimension int) [][]float64 {
	rand.Seed(time.Now().UnixNano())
	vectors := make([][]float64, count)
	for i := 0; i < count; i++ {
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rand.Float64()*2 - 1
		}
	}
	return vectors
}

func createTestVector(dimension int) []float64 {
	vector := make([]float64, dimension)
	for i := 0; i < dimension; i++ {
		vector[i] = rand.Float64()*2 - 1
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
