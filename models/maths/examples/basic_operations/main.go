package main

import (
	"fmt"
	"strings"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

func main() {
	fmt.Println("Maths Package - Basic Operations Example")
	fmt.Println(strings.Repeat("=", 50))

	// Basic arithmetic operations
	fmt.Println("\n1. Basic Arithmetic:")
	a, b := 10.5, 3.2
	fmt.Printf("Add(%f, %f) = %f\n", a, b, maths.Add(a, b))
	fmt.Printf("Subtract(%f, %f) = %f\n", a, b, maths.Subtract(a, b))
	fmt.Printf("Multiply(%f, %f) = %f\n", a, b, maths.Multiply(a, b))
	fmt.Printf("Divide(%f, %f) = %f\n", a, b, maths.Divide(a, b))
	fmt.Printf("Modulo(%f, %f) = %f\n", a, b, maths.Modulo(a, b))

	// Mathematical functions
	fmt.Println("\n2. Mathematical Functions:")
	x := 3.7
	fmt.Printf("Abs(%f) = %f\n", -x, maths.Abs(-x))
	fmt.Printf("Round(%f) = %f\n", x, maths.Round(x))
	fmt.Printf("Floor(%f) = %f\n", x, maths.Floor(x))
	fmt.Printf("Ceil(%f) = %f\n", x, maths.Ceil(x))

	// Statistical operations
	fmt.Println("\n3. Statistical Operations:")
	values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	fmt.Printf("Sum(%v) = %f\n", values, maths.Sum(values))
	fmt.Printf("Mean(%v) = %f\n", values, maths.Mean(values))
	fmt.Printf("Min(%v) = %f\n", values, maths.Min(values))
	fmt.Printf("Max(%v) = %f\n", values, maths.Max(values))

	// Comparison operations
	fmt.Println("\n4. Comparison Operations:")
	fmt.Printf("Equal(%f, %f) = %t\n", 3.14, 3.14, maths.Equal(3.14, 3.14))
	fmt.Printf("Greater(%f, %f) = %t\n", 5.0, 3.0, maths.Greater(5.0, 3.0))
	fmt.Printf("Less(%f, %f) = %t\n", 2.0, 4.0, maths.Less(2.0, 4.0))

	// Vector operations
	fmt.Println("\n5. Vector Operations:")
	vec1 := []float64{1.0, 2.0, 3.0, 4.0}
	vec2 := []float64{2.0, 3.0, 4.0, 5.0}

	cosine := maths.CosineAuto(vec1, vec2)
	fmt.Printf("Cosine similarity: %f\n", cosine)

	dot := maths.DotAuto(vec1, vec2)
	fmt.Printf("Dot product: %f\n", dot)

	// Performance monitoring
	fmt.Println("\n6. Performance Monitoring:")
	metrics := maths.GetPerformanceMetrics()
	fmt.Printf("Performance metrics: %+v\n", metrics)

	cacheStats := maths.GetCacheStats()
	fmt.Printf("Cache stats: %+v\n", cacheStats)

	// SIMD capabilities
	fmt.Println("\n7. SIMD Capabilities:")
	activePath := maths.GetActiveSIMDPath()
	fmt.Printf("Active SIMD path: %s\n", activePath)

	capabilities := maths.GetSIMDCapabilities()
	fmt.Printf("SIMD capabilities: %+v\n", capabilities)

	fmt.Println("\nExample completed successfully!")
}
