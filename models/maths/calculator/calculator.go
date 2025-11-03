// Package calculator provides a centralized calculator implementation for all agents
// This consolidates the duplicated calculator logic across Layer3 agents
package calculator

import (
	imaths "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

// Calculator provides mathematical operations for agent use
type Calculator struct{}

// NewCalculator creates a new calculator instance
func NewCalculator() *Calculator {
	return &Calculator{}
}

// Add returns a + b
func (c *Calculator) Add(a, b float64) float64 { 
	return imaths.Add(a, b) 
}

// Subtract returns a - b
func (c *Calculator) Subtract(a, b float64) float64 { 
	return imaths.Subtract(a, b) 
}

// Multiply returns a * b
func (c *Calculator) Multiply(a, b float64) float64 { 
	return imaths.Multiply(a, b) 
}

// Divide returns a / b
// Returns +Inf/-Inf if b is 0
func (c *Calculator) Divide(a, b float64) float64 { 
	return imaths.Divide(a, b) 
}

// Modulo returns a % b
func (c *Calculator) Modulo(a, b float64) float64 { 
	return imaths.Modulo(a, b) 
}

// Abs returns |a|
func (c *Calculator) Abs(a float64) float64 { 
	return imaths.Abs(a) 
}

// Equal returns true if a == b
func (c *Calculator) Equal(a, b float64) bool { 
	return imaths.Equal(a, b) 
}

// Greater returns true if a > b
func (c *Calculator) Greater(a, b float64) bool { 
	return imaths.Greater(a, b) 
}

// Less returns true if a < b
func (c *Calculator) Less(a, b float64) bool { 
	return imaths.Less(a, b) 
}

// Round returns a rounded to nearest integer
func (c *Calculator) Round(a float64) float64 { 
	return imaths.Round(a) 
}

// Floor returns the greatest integer less than or equal to a
func (c *Calculator) Floor(a float64) float64 { 
	return imaths.Floor(a) 
}

// Ceil returns the smallest integer greater than or equal to a
func (c *Calculator) Ceil(a float64) float64 { 
	return imaths.Ceil(a) 
}

// Sum returns the sum of all values
func (c *Calculator) Sum(values []float64) float64 { 
	return imaths.Sum(values) 
}

// Min returns the minimum value
func (c *Calculator) Min(values []float64) float64 { 
	return imaths.Min(values) 
}

// Max returns the maximum value
func (c *Calculator) Max(values []float64) float64 { 
	return imaths.Max(values) 
}

// Mean returns the arithmetic mean
func (c *Calculator) Mean(values []float64) float64 { 
	return imaths.Mean(values) 
}

// IntCalculator provides integer mathematical operations
type IntCalculator struct{}

// NewIntCalculator creates a new integer calculator instance
func NewIntCalculator() *IntCalculator {
	return &IntCalculator{}
}

// AddInt returns a + b
func (c *IntCalculator) AddInt(a, b int64) int64 { 
	return imaths.AddInt(a, b) 
}

// SubtractInt returns a - b
func (c *IntCalculator) SubtractInt(a, b int64) int64 { 
	return imaths.SubtractInt(a, b) 
}

// MultiplyInt returns a * b
func (c *IntCalculator) MultiplyInt(a, b int64) int64 { 
	return imaths.MultiplyInt(a, b) 
}

// DivideInt returns a / b
func (c *IntCalculator) DivideInt(a, b int64) int64 { 
	return imaths.DivideInt(a, b) 
}

// ModuloInt returns a % b
func (c *IntCalculator) ModuloInt(a, b int64) int64 { 
	return imaths.ModuloInt(a, b) 
}

// AbsInt returns |a|
func (c *IntCalculator) AbsInt(a int64) int64 { 
	return imaths.AbsInt(a) 
}

// EqualInt returns true if a == b
func (c *IntCalculator) EqualInt(a, b int64) bool { 
	return imaths.EqualInt(a, b) 
}

// GreaterInt returns true if a > b
func (c *IntCalculator) GreaterInt(a, b int64) bool { 
	return imaths.GreaterInt(a, b) 
}

// LessInt returns true if a < b
func (c *IntCalculator) LessInt(a, b int64) bool { 
	return imaths.LessInt(a, b) 
}

// SumInt returns the sum of all values
func (c *IntCalculator) SumInt(values []int64) int64 { 
	return imaths.SumInt(values) 
}

// MinInt returns the minimum value
func (c *IntCalculator) MinInt(values []int64) int64 { 
	return imaths.MinInt(values) 
}

// MaxInt returns the maximum value
func (c *IntCalculator) MaxInt(values []int64) int64 { 
	return imaths.MaxInt(values) 
}
