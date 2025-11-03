package util

// ----- Integer helpers (int64) -----

func AddInt(a, b int64) int64      { return a + b }
func SubtractInt(a, b int64) int64 { return a - b }
func MultiplyInt(a, b int64) int64 { return a * b }
func DivideInt(a, b int64) int64   { return a / b }
func ModuloInt(a, b int64) int64   { return a % b }

func AbsInt(a int64) int64 {
	if a < 0 {
		return -a
	}
	return a
}

func EqualInt(a, b int64) bool   { return a == b }
func GreaterInt(a, b int64) bool { return a > b }
func LessInt(a, b int64) bool    { return a < b }

func SumInt(values []int64) int64 {
	var acc int64
	for _, v := range values {
		acc += v
	}
	return acc
}

func MinInt(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] < m {
			m = values[i]
		}
	}
	return m
}

func MaxInt(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > m {
			m = values[i]
		}
	}
	return m
}
