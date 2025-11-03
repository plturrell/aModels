package stats

import (
	"fmt"
	"math"
	"sort"

	arraypkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/array"
)

// Array represents a statistical array
type Array = arraypkg.Array

// Percentile calculates the q-th percentile of the data
func Percentile(data []float64, q float64) (float64, error) {
	if q < 0 || q > 100 {
		return 0, fmt.Errorf("percentile must be between 0 and 100, got %f", q)
	}
	if len(data) == 0 {
		return 0, fmt.Errorf("cannot calculate percentile of empty data")
	}

	// Make a copy to avoid modifying the original
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	// Calculate index
	index := q / 100 * float64(len(sorted)-1)

	if index == float64(int(index)) {
		// Exact index
		return sorted[int(index)], nil
	}

	// Interpolate between adjacent values
	lower := int(index)
	upper := lower + 1
	weight := index - float64(lower)

	return sorted[lower]*(1-weight) + sorted[upper]*weight, nil
}

// Quantile calculates the q-th quantile (0 <= q <= 1)
func Quantile(data []float64, q float64) (float64, error) {
	if q < 0 || q > 1 {
		return 0, fmt.Errorf("quantile must be between 0 and 1, got %f", q)
	}
	return Percentile(data, q*100)
}

// Median calculates the median of the data
func Median(data []float64) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("cannot calculate median of empty data")
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2, nil
	}
	return sorted[n/2], nil
}

// Mode calculates the mode (most frequent value) of the data
func Mode(data []float64) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("cannot calculate mode of empty data")
	}

	// Count frequencies
	freq := make(map[float64]int)
	for _, v := range data {
		freq[v]++
	}

	// Find the most frequent value
	maxFreq := 0
	var mode float64
	for value, count := range freq {
		if count > maxFreq {
			maxFreq = count
			mode = value
		}
	}

	return mode, nil
}

// Variance calculates the variance of the data
func Variance(data []float64, ddof int) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("cannot calculate variance of empty data")
	}
	if ddof < 0 || ddof >= len(data) {
		return 0, fmt.Errorf("ddof must be between 0 and %d, got %d", len(data)-1, ddof)
	}

	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))

	sumSquares := 0.0
	for _, v := range data {
		diff := v - mean
		sumSquares += diff * diff
	}

	return sumSquares / float64(len(data)-ddof), nil
}

// StdDev calculates the standard deviation
func StdDev(data []float64, ddof int) (float64, error) {
	variance, err := Variance(data, ddof)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(variance), nil
}

// Covariance calculates the covariance between two datasets
func Covariance(x, y []float64, ddof int) (float64, error) {
	if len(x) != len(y) {
		return 0, fmt.Errorf("datasets must have the same length: %d vs %d", len(x), len(y))
	}
	if len(x) == 0 {
		return 0, fmt.Errorf("cannot calculate covariance of empty data")
	}
	if ddof < 0 || ddof >= len(x) {
		return 0, fmt.Errorf("ddof must be between 0 and %d, got %d", len(x)-1, ddof)
	}

	// Calculate means
	meanX := 0.0
	meanY := 0.0
	for i := 0; i < len(x); i++ {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(x))

	// Calculate covariance
	sum := 0.0
	for i := 0; i < len(x); i++ {
		sum += (x[i] - meanX) * (y[i] - meanY)
	}

	return sum / float64(len(x)-ddof), nil
}

// Correlation calculates the Pearson correlation coefficient
func Correlation(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0, fmt.Errorf("datasets must have the same length: %d vs %d", len(x), len(y))
	}
	if len(x) == 0 {
		return 0, fmt.Errorf("cannot calculate correlation of empty data")
	}

	// Calculate means
	meanX := 0.0
	meanY := 0.0
	for i := 0; i < len(x); i++ {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(x))

	// Calculate numerator and denominators
	num := 0.0
	sumX2 := 0.0
	sumY2 := 0.0

	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		num += dx * dy
		sumX2 += dx * dx
		sumY2 += dy * dy
	}

	if sumX2 == 0 || sumY2 == 0 {
		return 0, fmt.Errorf("one or both datasets have zero variance")
	}

	return num / math.Sqrt(sumX2*sumY2), nil
}

// Skewness calculates the skewness of the data
func Skewness(data []float64) (float64, error) {
	if len(data) < 3 {
		return 0, fmt.Errorf("need at least 3 data points for skewness")
	}

	// Calculate mean and standard deviation
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))

	stdDev, err := StdDev(data, 1)
	if err != nil {
		return 0, err
	}

	if stdDev == 0 {
		return 0, nil // All values are the same
	}

	// Calculate skewness
	sum := 0.0
	for _, v := range data {
		normalized := (v - mean) / stdDev
		sum += normalized * normalized * normalized
	}

	return sum / float64(len(data)), nil
}

// Kurtosis calculates the kurtosis of the data
func Kurtosis(data []float64) (float64, error) {
	if len(data) < 4 {
		return 0, fmt.Errorf("need at least 4 data points for kurtosis")
	}

	// Calculate mean and standard deviation
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))

	stdDev, err := StdDev(data, 1)
	if err != nil {
		return 0, err
	}

	if stdDev == 0 {
		return 0, nil // All values are the same
	}

	// Calculate kurtosis
	sum := 0.0
	for _, v := range data {
		normalized := (v - mean) / stdDev
		sum += normalized * normalized * normalized * normalized
	}

	return sum/float64(len(data)) - 3, nil // Excess kurtosis
}

// Array-based statistical helpers

func arrayDataCopy(a *Array) []float64 {
	return a.DataCopy()
}

// PercentileArray calculates percentiles for an array
func PercentileArray(a *Array, q float64) (float64, error) {
	return Percentile(arrayDataCopy(a), q)
}

// QuantileArray calculates quantiles for an array
func QuantileArray(a *Array, q float64) (float64, error) {
	return Quantile(arrayDataCopy(a), q)
}

// MedianArray calculates the median of an array
func MedianArray(a *Array) (float64, error) {
	return Median(arrayDataCopy(a))
}

// ModeArray calculates the mode of an array
func ModeArray(a *Array) (float64, error) {
	return Mode(arrayDataCopy(a))
}

// VarianceArray calculates the variance of an array
func VarianceArray(a *Array, ddof int) (float64, error) {
	return Variance(arrayDataCopy(a), ddof)
}

// StdDevArray calculates the standard deviation of an array
func StdDevArray(a *Array, ddof int) (float64, error) {
	return StdDev(arrayDataCopy(a), ddof)
}

// SkewnessArray calculates the skewness of an array
func SkewnessArray(a *Array) (float64, error) {
	return Skewness(arrayDataCopy(a))
}

// KurtosisArray calculates the kurtosis of an array
func KurtosisArray(a *Array) (float64, error) {
	return Kurtosis(arrayDataCopy(a))
}

// Statistical summary functions

// Describe returns a statistical summary of the data
type StatisticalSummary struct {
	Count    int
	Mean     float64
	StdDev   float64
	Min      float64
	Max      float64
	Median   float64
	Q25      float64
	Q75      float64
	Skewness float64
	Kurtosis float64
}

// DescribeArray returns a statistical summary of an array
func DescribeArray(a *Array) (*StatisticalSummary, error) {
	data := arrayDataCopy(a)
	if len(data) == 0 {
		return nil, fmt.Errorf("cannot describe empty array")
	}

	count := len(data)
	mean := a.Mean()
	stdDev, err := StdDevArray(a, 1)
	if err != nil {
		return nil, err
	}

	min := a.Min()
	max := a.Max()

	median, err := MedianArray(a)
	if err != nil {
		return nil, err
	}

	q25, err := QuantileArray(a, 0.25)
	if err != nil {
		return nil, err
	}

	q75, err := QuantileArray(a, 0.75)
	if err != nil {
		return nil, err
	}

	skewness, err := SkewnessArray(a)
	if err != nil {
		return nil, err
	}

	kurtosis, err := KurtosisArray(a)
	if err != nil {
		return nil, err
	}

	return &StatisticalSummary{
		Count:    count,
		Mean:     mean,
		StdDev:   stdDev,
		Min:      min,
		Max:      max,
		Median:   median,
		Q25:      q25,
		Q75:      q75,
		Skewness: skewness,
		Kurtosis: kurtosis,
	}, nil
}

// Rank calculates the rank of each element
func RankArray(a *Array) (*Array, error) {
	data := arrayDataCopy(a)
	if len(data) == 0 {
		return arraypkg.NewArray([]float64{}, 0), nil
	}

	// Create index-value pairs
	type pair struct {
		value float64
		index int
	}

	pairs := make([]pair, len(data))
	for i, v := range data {
		pairs[i] = pair{value: v, index: i}
	}

	// Sort by value
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value < pairs[j].value
	})

	// Assign ranks
	ranks := make([]float64, len(data))
	for i, pair := range pairs {
		ranks[pair.index] = float64(i + 1)
	}

	return arraypkg.NewArray(ranks, a.Shape()...), nil
}

// ZScore calculates the z-score (standardized values)
func ZScoreArray(a *Array) (*Array, error) {
	data := arrayDataCopy(a)
	if len(data) == 0 {
		return arraypkg.NewArray([]float64{}, 0), nil
	}

	mean := a.Mean()
	stdDev, err := StdDevArray(a, 1)
	if err != nil {
		return nil, err
	}

	if stdDev == 0 {
		zScores := make([]float64, len(data))
		return arraypkg.NewArray(zScores, a.Shape()...), nil
	}

	zScores := make([]float64, len(data))
	for i, v := range data {
		zScores[i] = (v - mean) / stdDev
	}

	return arraypkg.NewArray(zScores, a.Shape()...), nil
}

// Histogram calculates a histogram of the data
type Histogram struct {
	Counts []int
	Bins   []float64
}

// HistogramArray calculates a histogram of an array
func HistogramArray(a *Array, bins int) (*Histogram, error) {
	data := arrayDataCopy(a)
	if len(data) == 0 {
		return nil, fmt.Errorf("cannot calculate histogram of empty array")
	}
	if bins <= 0 {
		return nil, fmt.Errorf("number of bins must be positive, got %d", bins)
	}

	// Find min and max
	min := a.Min()
	max := a.Max()

	if min == max {
		// All values are the same
		counts := make([]int, bins)
		counts[0] = len(data)
		binEdges := make([]float64, bins+1)
		for i := range binEdges {
			binEdges[i] = min
		}
		return &Histogram{Counts: counts, Bins: binEdges}, nil
	}

	// Create bins
	binWidth := (max - min) / float64(bins)
	binEdges := make([]float64, bins+1)
	for i := 0; i <= bins; i++ {
		binEdges[i] = min + float64(i)*binWidth
	}

	// Count values in each bin
	counts := make([]int, bins)
	for _, v := range data {
		binIndex := int((v - min) / binWidth)
		if binIndex >= bins {
			binIndex = bins - 1 // Put max value in last bin
		}
		counts[binIndex]++
	}

	return &Histogram{Counts: counts, Bins: binEdges}, nil
}
