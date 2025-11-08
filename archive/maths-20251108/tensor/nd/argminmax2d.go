package nd

// ArgMinAxis returns indices of minimum values along an axis for 2D slice.
func ArgMinAxis(A [][]float64, axis int) []int {
    if axis == 0 {
        n := len(A[0])
        result := make([]int, n)
        for j := 0; j < n; j++ {
            minIdx := 0
            minVal := A[0][j]
            for i := 1; i < len(A); i++ { if A[i][j] < minVal { minVal = A[i][j]; minIdx = i } }
            result[j] = minIdx
        }
        return result
    }
    m := len(A)
    result := make([]int, m)
    for i := 0; i < m; i++ {
        minIdx := 0
        minVal := A[i][0]
        for j := 1; j < len(A[i]); j++ { if A[i][j] < minVal { minVal = A[i][j]; minIdx = j } }
        result[i] = minIdx
    }
    return result
}

// ArgMaxAxis returns indices of maximum values along an axis for 2D slice.
func ArgMaxAxis(A [][]float64, axis int) []int {
    if axis == 0 {
        n := len(A[0])
        result := make([]int, n)
        for j := 0; j < n; j++ {
            maxIdx := 0
            maxVal := A[0][j]
            for i := 1; i < len(A); i++ { if A[i][j] > maxVal { maxVal = A[i][j]; maxIdx = i } }
            result[j] = maxIdx
        }
        return result
    }
    m := len(A)
    result := make([]int, m)
    for i := 0; i < m; i++ {
        maxIdx := 0
        maxVal := A[i][0]
        for j := 1; j < len(A[i]); j++ { if A[i][j] > maxVal { maxVal = A[i][j]; maxIdx = j } }
        result[i] = maxIdx
    }
    return result
}

// ArgSort returns indices that would sort the array
func ArgSort(data []float64) []int {
    n := len(data)
    indices := make([]int, n)
    for i := 0; i < n; i++ { indices[i] = i }
    quicksortIndices(data, indices, 0, n-1)
    return indices
}

func quicksortIndices(data []float64, indices []int, low, high int) {
    if low < high {
        pivot := partitionIndices(data, indices, low, high)
        quicksortIndices(data, indices, low, pivot-1)
        quicksortIndices(data, indices, pivot+1, high)
    }
}

func partitionIndices(data []float64, indices []int, low, high int) int {
    pivot := data[indices[high]]
    i := low - 1
    for j := low; j < high; j++ {
        if data[indices[j]] <= pivot { i++; indices[i], indices[j] = indices[j], indices[i] }
    }
    indices[i+1], indices[high] = indices[high], indices[i+1]
    return i + 1
}

