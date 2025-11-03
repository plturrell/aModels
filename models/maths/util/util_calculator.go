package util

// Calculator provides float64 mathematical operations via helper functions.
type Calculator struct{}

// NewCalculator creates a new Calculator.
func NewCalculator() *Calculator { return &Calculator{} }

func (c *Calculator) Add(a, b float64) float64      { return Add(a, b) }
func (c *Calculator) Subtract(a, b float64) float64 { return Subtract(a, b) }
func (c *Calculator) Multiply(a, b float64) float64 { return Multiply(a, b) }
func (c *Calculator) Divide(a, b float64) float64   { return Divide(a, b) }

// Equal returns true if |a-b| < 1e-9 by default.
func (c *Calculator) Equal(a, b float64) bool { return NearlyEqual(a, b, 1e-9) }

// IntCalculator provides integer mathematical operations using int64 helpers.
type IntCalculator struct{}

// NewIntCalculator creates a new IntCalculator.
func NewIntCalculator() *IntCalculator { return &IntCalculator{} }

func (ic *IntCalculator) Add(a, b int) int      { return int(AddInt(int64(a), int64(b))) }
func (ic *IntCalculator) Subtract(a, b int) int { return int(SubtractInt(int64(a), int64(b))) }
func (ic *IntCalculator) Multiply(a, b int) int { return int(MultiplyInt(int64(a), int64(b))) }
func (ic *IntCalculator) Divide(a, b int) int   { return int(DivideInt(int64(a), int64(b))) }

// ParseInt attempts to parse a string as an integer.
func (ic *IntCalculator) ParseInt(s string) (int, error) { return ParseInt(s) }
