package tensor

import (
    numa "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/numa"
    poolpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/pool"
    rtpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/runtime"
    quantpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/quant"
)

// ============================================================================
// EXTREME CPU OPTIMIZATIONS - Faster Than TensorFlow GPU
// Target: Beat GPU performance using only CPU
// ============================================================================

// ExtremeConfig holds extreme optimization settings
type ExtremeConfig = numa.ExtremeConfig
func DetectExtremeCapabilities() *ExtremeConfig { return numa.DetectExtremeCapabilities() }

// ============================================================================
// AVX-512 OPTIMIZED MATRIX MULTIPLICATION
// Process 16 floats at once (512 bits / 32 bits = 16)
// ============================================================================

// AVX512MatMul performs AVX-512 optimized matrix multiplication
// Target: 0.5ms for 1024×1024 (GPU-level performance)
func (t *TensorOps) AVX512MatMul(A, B [][]float64) ([][]float64, error) { return numa.AVX512MatMul(A,B) }

// ============================================================================
// NUMA-AWARE MEMORY ALLOCATION
// Distribute data across CPU sockets for multi-socket systems
// ============================================================================

// NUMAMatrix represents a NUMA-aware matrix
type NUMAMatrix = numa.NUMAMatrix
func NewNUMAMatrix(m, n, sockets int) *NUMAMatrix { return numa.NewNUMAMatrix(m,n,sockets) }

// NUMAMatMul performs NUMA-aware matrix multiplication
func (t *TensorOps) NUMAMatMul(A, B *NUMAMatrix) (*NUMAMatrix, error) { return numa.NUMAMatMul(A,B) }

// ============================================================================
// KERNEL FUSION - Eliminate Memory Transfers
// Fuse multiple operations into single kernel
// ============================================================================

// FusedConvBNReLU fuses Conv2D + BatchNorm + ReLU
// Eliminates 2 memory transfers (GPU-level fusion)
func (t *TensorOps) FusedConvBNReLU(input, kernel [][]float64, gamma, beta []float64, eps float64) ([][]float64, error) { return numa.FusedConvBNReLU(input, kernel, gamma, beta, eps) }

// FusedLinearLayerForward fuses Linear + Bias + Activation
func (t *TensorOps) FusedLinearLayerForward(input, weight [][]float64, bias []float64, activation string) ([][]float64, error) { return numa.FusedLinearLayerForward(input, weight, bias, activation) }

// ============================================================================
// INT8 QUANTIZATION - delegated to quant subpackage
type QuantizedMatrix = quantpkg.QuantizedMatrix
func QuantizeMatrix(A [][]float64) *QuantizedMatrix { return quantpkg.QuantizeMatrix(A) }
func (t *TensorOps) QuantizedMatMul(A, B *QuantizedMatrix) (*QuantizedMatrix, error) { return quantpkg.QuantizedMatMul(A,B) }

// ============================================================================
// PREFETCHING AND CACHE OPTIMIZATION
// ============================================================================

// Prefetching and cache optimization
func PrefetchMatrix(A [][]float64, startRow, endRow int) { numa.PrefetchMatrix(A, startRow, endRow) }

// CacheBlockedMatMul with explicit prefetching
func (t *TensorOps) CacheBlockedMatMulPrefetch(A, B [][]float64) ([][]float64, error) { return numa.CacheBlockedMatMulPrefetch(A, B, t.parallelCfg) }

// ============================================================================
// LOCK-FREE PARALLEL REDUCTION
// ============================================================================

// Thin wrapper to runtime subpackage
func LockFreeSum(data []float64) float64 { return rtpkg.LockFreeSum(data) }

// ============================================================================
// MEMORY POOL FOR ZERO-ALLOCATION
// ============================================================================

// Re-export MatrixPool from pool subpackage
type MatrixPool = poolpkg.MatrixPool
func NewMatrixPool() *MatrixPool { return poolpkg.NewMatrixPool() }

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// (removed unused helper)
