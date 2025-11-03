package maths

import (
    "math"
    "os"
    "strconv"
    "time"

    back "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/backend"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/cache"
    gemmpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/gemm"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/monitoring"
    simdpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/simd"
    tensorpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor"
    util "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// MatMul2D forwards to the packed GEMM implementation.
func MatMul2D(A, B [][]float64) [][]float64 { return gemmpkg.MatMul2D(A, B) }

// NewTensorOps forwards to tensor package constructor.
func NewTensorOps(useFortran bool) *tensorpkg.TensorOps { return tensorpkg.NewTensorOps(useFortran) }

// NewTensorOpsWithThreads forwards to tensor package constructor with explicit threads.
func NewTensorOpsWithThreads(useFortran bool, threads int) *tensorpkg.TensorOps {
	return tensorpkg.NewTensorOpsWithThreads(useFortran, threads)
}

// SetSIMDAccuracyMode forwards to tensor package.
func SetSIMDAccuracyMode(mode tensorpkg.SIMDAccuracyMode) { tensorpkg.SetSIMDAccuracyMode(mode) }

// Export accuracy mode constants.
const (
	SIMDFastApproximate = tensorpkg.SIMDFastApproximate
	SIMDStrictAccuracy  = tensorpkg.SIMDStrictAccuracy
)

// Transcendental helpers (forwarders)
func TanhEstrinF64(x float64) float64          { return tensorpkg.TanhEstrinF64(x) }
func Log1pEstrinF64(x float64) float64         { return tensorpkg.Log1pEstrinF64(x) }
func Expm1EstrinF64(x float64) float64         { return tensorpkg.Expm1EstrinF64(x) }
func TanhEstrinF32(x float32) float32          { return tensorpkg.TanhEstrinF32(x) }
func VectorizedTanhF64(x []float64) []float64  { return tensorpkg.VectorizedTanhF64(x) }
func VectorizedLog1pF64(x []float64) []float64 { return tensorpkg.VectorizedLog1pF64(x) }
func VectorizedExpm1F64(x []float64) []float64 { return tensorpkg.VectorizedExpm1F64(x) }
func VectorizedTanhF32(x []float32) []float32  { return tensorpkg.VectorizedTanhF32(x) }

// Fused ops (forwarders)
func FusedSoftmaxCrossEntropy(logits [][]float64, labels []int) (float64, [][]float64) {
	return tensorpkg.SoftmaxCrossEntropy(logits, labels)
}

// Float32 softmax helpers (additive wrappers)
func SoftmaxRow32(x []float32) []float32    { return tensorpkg.SoftmaxRow32(x) }
func Softmax2D32(X [][]float32) [][]float32 { return tensorpkg.Softmax2D32(X) }
func SoftmaxCrossEntropy32(logits [][]float32, labels []int) (float32, [][]float32) {
	return tensorpkg.SoftmaxCrossEntropy32(logits, labels)
}

// Float64 softmax helpers (additive wrappers)
func SoftmaxRow64(x []float64) []float64    { return tensorpkg.SoftmaxRow64(x) }
func Softmax2D64(X [][]float64) [][]float64 { return tensorpkg.Softmax2D64(X) }

// Float32 FlashAttention helper
func FlashAttention2D32(Q, K, V [][]float32, scale float32) [][]float32 {
	return tensorpkg.FlashAttention2D32(Q, K, V, scale)
}

// SIMD helper forwarders used by other packages
func SIMDFusedMultiplyAddF32(a, b, c []float32) []float32 {
	return simdpkg.SIMDFusedMultiplyAddF32(a, b, c)
}

// --- Legacy/agent-facing compatibility layer ---

// Provider mirrors backend.Provider so agent code can call maths.New().Cos(...)
type Provider = back.Provider

// New selects the active maths backend (Go or Fortran if registered).
func New() Provider { return back.New() }

// Random projection builders and projectors
func BuildRandomProjection(n, r int, seed int64) []float64 {
	return back.BuildRandomProjection(n, r, seed)
}
func Project(m, n, r int, A, P []float64) []float64 { return back.Project(m, n, r, A, P) }

// Cosine helpers
func CosineTopK(n int, A []float64, q []float64, topK int) ([]int, []float64) {
	return back.CosineTopK(n, A, q, topK)
}
func CosineMultiTopK(n int, A []float64, Q []float64, topK int) ([][]int, [][]float64) {
	return back.CosineMultiTopK(n, A, Q, topK)
}
func CosineTopKInt8(n int, A8 []int8, q []float64, topK int) ([]int, []float64) {
	return back.CosineTopKInt8(n, A8, q, topK)
}
func CosineAuto(a, b []float64) float64 {
	// Check cache first
	cache := cache.GetGlobalCache()
	key := cache.VectorCacheKey(a, b)
	if result, ok := cache.Get(key); ok {
		return result.(float64)
	}

	// Compute and cache result
	start := time.Now()
	result := back.CosineAuto(a, b)
	duration := time.Since(start)

	// Record performance metrics
	dashboard := monitoring.GetGlobalDashboard()
	dashboard.RecordOperation("cosine", duration, nil, 0)

	cache.Put(key, result)
	return result
}

func CosineBatchAuto(n int, A, B []float64) []float64 {
	// For batch operations, we don't cache individual results
	// but we do record performance metrics
	start := time.Now()
	result := back.CosineBatchAuto(n, A, B)
	duration := time.Since(start)

	// Record performance metrics
	dashboard := monitoring.GetGlobalDashboard()
	dashboard.RecordOperation("cosine_batch", duration, nil, 0)

	return result
}

// Dot helpers
func DotAuto(a, b []float64) float64 {
	// Check cache first
	cache := cache.GetGlobalCache()
	key := cache.VectorCacheKey(a, b)
	if result, ok := cache.Get(key); ok {
		return result.(float64)
	}

	// Compute and cache result
	start := time.Now()
	result := back.DotAuto(a, b)
	duration := time.Since(start)

	// Record performance metrics
	dashboard := monitoring.GetGlobalDashboard()
	dashboard.RecordOperation("dot", duration, nil, 0)

	cache.Put(key, result)
	return result
}

func DotBatchAuto(n int, A, B []float64) []float64 {
	// For batch operations, we don't cache individual results
	// but we do record performance metrics
	start := time.Now()
	result := back.DotBatchAuto(n, A, B)
	duration := time.Since(start)

	// Record performance metrics
	dashboard := monitoring.GetGlobalDashboard()
	dashboard.RecordOperation("dot_batch", duration, nil, 0)

	return result
}
func Dot(a, b []float64) float64 { return back.New().Dot(a, b) }

// Flat MatMul: C(m x n) = A(m x k) * B(k x n), row-major
func MatMul(m, n, k int, A, B []float64) []float64 { return back.New().MatMul(m, n, k, A, B) }

// Fixed-point (Q16) helpers
func Q16FromFloat(x float64) int32 { return util.Q16FromFloat(x) }
func Q16ToFloat(q int32) float64   { return util.Q16ToFloat(q) }

// Parsing helpers
func ParseInt(s string) (int, error)       { return util.ParseInt(s) }
func ParseFloat(s string) (float64, error) { return util.ParseFloat(s) }

// Acceleration configuration (compat shim)
type AccelConfig struct {
	EnableFortran bool
	Threads       int // optional: desired worker threads
}

// SetAccelConfig influences maths backend selection via environment.
// EnableFortran=false forces pure Go; true prefers registered Fortran provider if present.
func SetAccelConfig(cfg AccelConfig) {
	if cfg.EnableFortran {
		os.Setenv("INFRA_MATHS_BACKEND", "fortran")
	} else {
		os.Setenv("INFRA_MATHS_BACKEND", "go")
	}
	if cfg.Threads > 0 {
		// Hints; backends may respect these.
		os.Setenv("GOMAXPROCS", strconv.Itoa(cfg.Threads))
		os.Setenv("OMP_NUM_THREADS", strconv.Itoa(cfg.Threads))
		os.Setenv("OPENBLAS_NUM_THREADS", strconv.Itoa(cfg.Threads))
		os.Setenv("MKL_NUM_THREADS", strconv.Itoa(cfg.Threads))
	}
}

// TensorOps type alias for agent code expecting maths.TensorOps
type TensorOps = tensorpkg.TensorOps

// Fusions helper used by agents
func FusedAddMulExp(a, b, c []float64) []float64 { return tensorpkg.AddMulExp(a, b, c) }

// Simple Sqrt forwarder for agent convenience
func Sqrt(x float64) float64 { return math.Sqrt(x) }

// Scalar float helpers (compat)
func Add(a, b float64) float64      { return util.Add(a, b) }
func Subtract(a, b float64) float64 { return util.Subtract(a, b) }
func Multiply(a, b float64) float64 { return util.Multiply(a, b) }
func Divide(a, b float64) float64   { return util.Divide(a, b) }
func Modulo(a, b float64) float64   { return util.Modulo(a, b) }
func Abs(a float64) float64         { return util.Abs(a) }
func Equal(a, b float64) bool       { return util.Equal(a, b) }
func Greater(a, b float64) bool     { return util.Greater(a, b) }
func Less(a, b float64) bool        { return util.Less(a, b) }
func Round(a float64) float64       { return util.Round(a) }
func Floor(a float64) float64       { return util.Floor(a) }
func Ceil(a float64) float64        { return util.Ceil(a) }
func Sum(values []float64) float64  { return util.Sum(values) }
func Min(values []float64) float64  { return util.Min(values) }
func Max(values []float64) float64  { return util.Max(values) }
func Mean(values []float64) float64 { return util.Mean(values) }

// Integer helpers (compat: use int64 to match agent wrappers)
func AddInt(a, b int64) int64      { return util.AddInt(a, b) }
func SubtractInt(a, b int64) int64 { return util.SubtractInt(a, b) }
func MultiplyInt(a, b int64) int64 { return util.MultiplyInt(a, b) }
func DivideInt(a, b int64) int64   { return util.DivideInt(a, b) }
func ModuloInt(a, b int64) int64   { return util.ModuloInt(a, b) }
func AbsInt(a int64) int64         { return util.AbsInt(a) }
func EqualInt(a, b int64) bool     { return util.EqualInt(a, b) }
func GreaterInt(a, b int64) bool   { return util.GreaterInt(a, b) }
func LessInt(a, b int64) bool      { return util.LessInt(a, b) }
func SumInt(values []int64) int64  { return util.SumInt(values) }
func MinInt(values []int64) int64  { return util.MinInt(values) }
func MaxInt(values []int64) int64  { return util.MaxInt(values) }

// Calculator re-exports
type Calculator = util.Calculator
type IntCalculator = util.IntCalculator

func NewCalculator() *Calculator       { return util.NewCalculator() }
func NewIntCalculator() *IntCalculator { return util.NewIntCalculator() }

// --- Enterprise Features ---

// GetPerformanceMetrics returns current performance metrics
func GetPerformanceMetrics() map[string]interface{} {
	dashboard := monitoring.GetGlobalDashboard()
	return dashboard.GetMetrics()
}

// GetCacheStats returns cache statistics
func GetCacheStats() map[string]interface{} {
	cache := cache.GetGlobalCache()
	return cache.Stats()
}

// ClearCache clears the global cache
func ClearCache() {
	cache := cache.GetGlobalCache()
	cache.Clear()
}

// SetCacheSize sets the global cache size
func SetCacheSize(size int) {
	cache.SetGlobalCacheSize(size)
}

// GetTopOperations returns the most frequently used operations
func GetTopOperations(limit int) []map[string]interface{} {
	dashboard := monitoring.GetGlobalDashboard()
	return dashboard.GetTopOperations(limit)
}

// GetBottlenecks returns performance bottlenecks
func GetBottlenecks() []string {
	dashboard := monitoring.GetGlobalDashboard()
	return dashboard.GetBottlenecks()
}

// GetOperationHeatmap returns operation performance heatmap
func GetOperationHeatmap() map[string]interface{} {
	dashboard := monitoring.GetGlobalDashboard()
	return dashboard.GetOperationHeatmap()
}

// GetActiveSIMDPath returns the currently active SIMD instruction set
func GetActiveSIMDPath() string {
	// Check for AVX-512 support
	if simdpkg.HasAVX512() {
		return "AVX-512"
	}
	// Check for AVX2 support
	if simdpkg.HasAVX2() {
		return "AVX2"
	}
	// Check for NEON support (ARM64)
	if simdpkg.HasNEON() {
		return "NEON"
	}
	// Fallback to scalar
	return "Scalar"
}

// GetSIMDCapabilities returns all available SIMD capabilities
func GetSIMDCapabilities() map[string]bool {
	return map[string]bool{
		"AVX-512": simdpkg.HasAVX512(),
		"AVX2":    simdpkg.HasAVX2(),
		"NEON":    simdpkg.HasNEON(),
		"SSE4":    simdpkg.HasSSE4(),
	}
}
