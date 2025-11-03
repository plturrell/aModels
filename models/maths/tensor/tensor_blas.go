package tensor

import (
    "fmt"
    linalg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/linalg"
    poolpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/pool"
)

// ============================================================================
// OPTIONAL BLAS INTEGRATION (macOS Accelerate, OpenBLAS, MKL)
// Falls back to pure Go if BLAS unavailable
// ============================================================================

// Re-export BLAS types and helpers from linalg to keep API stable
type BLASBackend = linalg.BLASBackend
type BLASConfig = linalg.BLASConfig
func InitBLAS(numThreads int) *BLASConfig { return linalg.InitBLAS(numThreads) }
func GetBLASInfo() string { return linalg.GetBLASInfo() }

// ============================================================================
// OPTIMIZED EIGENDECOMPOSITION (DSYEVD - Divide and Conquer)
// ============================================================================

// EigenDecompositionDC performs divide-and-conquer eigendecomposition
// Faster than power iteration for larger matrices (n > 32)
// Falls back to power iteration if BLAS unavailable
func (t *TensorOps) EigenDecompositionDC(A [][]float64) ([]float64, [][]float64, error) {
    n := len(A)
    if n == 0 {
        return nil, nil, fmt.Errorf("empty matrix")
    }

    config := linalg.InitBLAS(t.numThreads)

    // Optional override via env: MATHS_EIGEN_PREFERRED=gonum|dc|power
    switch linalg.PreferredEigenImpl() {
    case "gonum", "blas", "dsyevd":
        return linalg.EigenDecompositionBLAS(A)
    case "dc", "divide", "divideconquer":
        return linalg.EigenDecompositionDivideConquer(A)
    case "power":
        return linalg.EigenDecomposition(A)
    }

    // Default path: prefer BLAS/Gonum for larger n
    if config.Available && n > 32 {
        return linalg.EigenDecompositionBLAS(A)
    }

    // Fall back to pure Go power iteration
    return linalg.EigenDecomposition(A)
}

// eigenDecompositionBLAS uses BLAS dsyevd (divide-and-conquer)
// kept for compatibility; delegate to linalg package
func eigenDecompositionBLAS(A [][]float64, _ *BLASConfig) ([]float64, [][]float64, error) { return linalg.EigenDecompositionBLAS(A) }

// eigenDecompositionDivideConquer implements divide-and-conquer for symmetric matrices
func eigenDecompositionDivideConquer(A [][]float64) ([]float64, [][]float64, error) {
    return linalg.EigenDecompositionDivideConquer(A)
}

// eigenDecompositionDirect solves small matrices directly
func eigenDecompositionDirect(A [][]float64) ([]float64, [][]float64, error) { return linalg.EigenDecomposition(A) }

// ============================================================================
// BUFFER REUSE FOR PERFORMANCE (moved to tensor/pool)
// ============================================================================

// Re-export types and constructors to preserve tensor API
type BufferPool = poolpkg.BufferPool
func NewBufferPool() *BufferPool { return poolpkg.NewBufferPool() }

type MatrixBufferPool = poolpkg.MatrixBufferPool
func NewMatrixBufferPool() *MatrixBufferPool { return poolpkg.NewMatrixBufferPool() }

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// (extractSubmatrix moved to linalg package; no longer needed here)
