package tensor

import (
	act "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/activations"
	alg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/algorithms"
	arraypkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/array"
	attn "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/attention"
	convpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/conv"
	fftpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/fft"
	f "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/fusions"
	lane "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/lanes"
	linalgpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/linalg"
	ops "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/ops"
	q "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/quant"
	sp "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/sparse"
	utilpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// ===== Activations / Transcendentals =====

type SIMDAccuracyMode = act.SIMDAccuracyMode

const (
	SIMDFastApproximate = act.SIMDFastApproximate
	SIMDStrictAccuracy  = act.SIMDStrictAccuracy
)

func SetSIMDAccuracyMode(mode SIMDAccuracyMode) { act.SetSIMDAccuracyMode(mode) }

func TanhEstrinF64(x float64) float64  { return act.TanhEstrinF64(x) }
func Log1pEstrinF64(x float64) float64 { return act.Log1pEstrinF64(x) }
func Expm1EstrinF64(x float64) float64 { return act.Expm1EstrinF64(x) }
func TanhEstrinF32(x float32) float32  { return act.TanhEstrinF32(x) }

func VectorizedTanhF64(x []float64) []float64  { return act.VectorizedTanhF64(x) }
func VectorizedLog1pF64(x []float64) []float64 { return act.VectorizedLog1pF64(x) }
func VectorizedExpm1F64(x []float64) []float64 { return act.VectorizedExpm1F64(x) }
func VectorizedTanhF32(x []float32) []float32  { return act.VectorizedTanhF32(x) }

// ===== Attention =====

func FlashAttention(Q, K, V *utilpkg.Matrix64, scale float64) *utilpkg.Matrix64 {
	return attn.FlashAttention(Q, K, V, scale)
}

func FlashAttention2D(Q, K, V [][]float64, scale float64) [][]float64 {
	return attn.FlashAttention2D(Q, K, V, scale)
}

// Float32 FlashAttention
func FlashAttention2D32(Q, K, V [][]float32, scale float32) [][]float32 {
	return attn.FlashAttention2D32(Q, K, V, scale)
}

// ===== FFT =====

type FFTConfig = fftpkg.FFTConfig
type FFTAxisConfig = fftpkg.FFTAxisConfig

func DefaultFFTConfig() *FFTConfig                 { return fftpkg.DefaultFFTConfig() }
func DefaultFFTAxisConfig(axis int) *FFTAxisConfig { return fftpkg.DefaultFFTAxisConfig(axis) }

func FFT(x []complex128, cfg *FFTConfig) []complex128  { return fftpkg.FFT(x, cfg) }
func IFFT(x []complex128, cfg *FFTConfig) []complex128 { return fftpkg.IFFT(x, cfg) }
func RFFT(x []float64, cfg *FFTConfig) []complex128    { return fftpkg.RFFT(x, cfg) }
func FFT2DAxis(x [][]complex128, axis int, cfg *FFTAxisConfig) [][]complex128 {
	return fftpkg.FFT2DAxis(x, axis, cfg)
}
func IFFT2DAxis(x [][]complex128, axis int, cfg *FFTAxisConfig) [][]complex128 {
	return fftpkg.IFFT2DAxis(x, axis, cfg)
}
func FFT3DAxis(x [][][]complex128, axis int, cfg *FFTAxisConfig) [][][]complex128 {
	return fftpkg.FFT3DAxis(x, axis, cfg)
}
func IFFT3DAxis(x [][][]complex128, axis int, cfg *FFTAxisConfig) [][][]complex128 {
	return fftpkg.IFFT3DAxis(x, axis, cfg)
}
func FFTNDAxes(x [][][]complex128, axes []int, cfg *FFTConfig) [][][]complex128 {
	return fftpkg.FFTNDAxes(x, axes, cfg)
}
func IFFTNDAxes(x [][][]complex128, axes []int, cfg *FFTConfig) [][][]complex128 {
	return fftpkg.IFFTNDAxes(x, axes, cfg)
}

// ===== SIMD lanes =====

func LaneAddF32_AVX512(a, b []float32) []float32    { return lane.LaneAddF32_AVX512(a, b) }
func LaneMulF32_AVX512(a, b []float32) []float32    { return lane.LaneMulF32_AVX512(a, b) }
func LaneFMAF32_AVX512(a, b, c []float32) []float32 { return lane.LaneFMAF32_AVX512(a, b, c) }
func LaneDotF32_NEON(a, b []float32) float32        { return lane.LaneDotF32_NEON(a, b) }
func SelectOptimalF32Add(a, b []float32) []float32  { return lane.SelectOptimalF32Add(a, b) }

// ===== Fusions (Softmax) float32 =====

func SoftmaxRow32(x []float32) []float32    { return f.SoftmaxRow32(x) }
func Softmax2D32(X [][]float32) [][]float32 { return f.Softmax2D32(X) }
func SoftmaxCrossEntropy32(logits [][]float32, labels []int) (float32, [][]float32) {
	return f.SoftmaxCrossEntropy32(logits, labels)
}

// ===== Fusions (Softmax) float64 =====

func SoftmaxRow64(x []float64) []float64    { return f.SoftmaxRow64(x) }
func Softmax2D64(X [][]float64) [][]float64 { return f.Softmax2D64(X) }
func SoftmaxCrossEntropy(logits [][]float64, labels []int) (float64, [][]float64) {
	return f.SoftmaxCrossEntropy(logits, labels)
}

// Fused elementwise utilities
func AddMulExp(a, b, c []float64) []float64 { return f.AddMulExp(a, b, c) }

// ===== Lane utilities =====

// ===== ND tensors =====

// (ndpkg no longer exports typed ND arrays; high-level helpers removed.)

// ===== Ops wrappers =====

func (t *TensorOps) Einsum(subscripts string, operands ...interface{}) (interface{}, error) {
	return ops.Einsum(subscripts, operands...)
}
func (t *TensorOps) Transpose(A [][]float64) [][]float64 { return ops.Transpose(A) }
func (t *TensorOps) Reshape(data []float64, shape ...int) (interface{}, error) {
	return ops.Reshape(data, shape...)
}
func (t *TensorOps) Broadcast(A, B [][]float64) ([][]float64, [][]float64, error) {
	return ops.Broadcast(A, B)
}
func (t *TensorOps) Permute(A [][][]float64, dims []int) ([][][]float64, error) {
	return ops.Permute(A, dims)
}

// ===== Sparse + Quant wrappers =====

type SparseCOO = sp.SparseCOO
type SparseCSR = sp.SparseCSR

func (t *TensorOps) NewSparseCOO(dense [][]float64, threshold float64) *SparseCOO {
	return sp.NewSparseCOO(dense, threshold)
}
func (t *TensorOps) COOToCSR(coo *SparseCOO) *SparseCSR { return sp.COOToCSR(coo) }
func (t *TensorOps) SparseMatVec(csr *SparseCSR, x []float64) ([]float64, error) {
	return sp.SparseMatVec(csr, x)
}
func (t *TensorOps) SparseMatMul(A, B *SparseCSR) (*SparseCSR, error) {
	return sp.SparseMatMul(A, B)
}
func (t *TensorOps) SparseToDense(csr *SparseCSR) [][]float64 { return sp.SparseToDense(csr) }

type Float16 = q.Float16

func (t *TensorOps) MixedPrecisionGEMM(A, B [][]float64) ([][]float64, error) {
	return q.MixedPrecisionGEMM(A, B)
}
func Float32ToFloat16(f float32) Float16                                   { return q.Float32ToFloat16(f) }
func Float16ToFloat32(h Float16) float32                                   { return q.Float16ToFloat32(h) }
func (t *TensorOps) QuantizeInt8(A [][]float64) ([][]int8, float64, error) { return q.QuantizeInt8(A) }
func (t *TensorOps) DequantizeInt8(quantized [][]int8, scale float64) [][]float64 {
	return q.DequantizeInt8(quantized, scale)
}
func (t *TensorOps) Int8MatMul(A, B [][]int8, scaleA, scaleB float64) ([][]float64, error) {
	return q.Int8MatMul(A, B, scaleA, scaleB)
}

// ===== Array (NumPy-like) =====

type Array = arraypkg.Array

func NewArray(data []float64, shape ...int) *Array { return arraypkg.NewArray(data, shape...) }
func Zeros(shape ...int) *Array                    { return arraypkg.Zeros(shape...) }
func Ones(shape ...int) *Array                     { return arraypkg.Ones(shape...) }
func Arange(start, stop, step float64) *Array      { return arraypkg.Arange(start, stop, step) }
func Linspace(start, stop float64, num int) *Array { return arraypkg.Linspace(start, stop, num) }
func Eye(n int) *Array                             { return arraypkg.Eye(n) }
func Random(shape ...int) *Array                   { return arraypkg.Random(shape...) }
func Concatenate(arrays []*Array, axis int) *Array {
	a := make([]*arraypkg.Array, len(arrays))
	for i := range arrays {
		a[i] = (*arraypkg.Array)(arrays[i])
	}
	return arraypkg.Concatenate(a, axis)
}
func Stack(arrays []*Array, axis int) *Array {
	a := make([]*arraypkg.Array, len(arrays))
	for i := range arrays {
		a[i] = (*arraypkg.Array)(arrays[i])
	}
	return arraypkg.Stack(a, axis)
}
func Solve(A, b *Array) (*Array, error) {
	res, err := arraypkg.Solve((*arraypkg.Array)(A), (*arraypkg.Array)(b))
	return (*Array)(res), err
}

// ===== Algorithms / Convolution / Linalg wrappers =====

func (t *TensorOps) StrassenMatMul(A, B [][]float64) ([][]float64, error) {
	return alg.StrassenMatMul(A, B)
}

// WinogradConv2D delegates to conv subpackage for centralized conv impl
func (t *TensorOps) WinogradConv2D(input [][]float64, kernel [][]float64, stride int) ([][]float64, error) {
	return convpkg.WinogradConv2D(input, kernel, stride)
}

// QR/SVD delegates to linalg package to keep root lean
func (t *TensorOps) QRDecomposition(A [][]float64) (Q, R [][]float64, err error) {
	return linalgpkg.QRDecomposition(A)
}
func (t *TensorOps) SVD(A [][]float64, fullMatrices bool) (U, S [][]float64, Vt [][]float64, err error) {
	return linalgpkg.SVD(A, fullMatrices)
}
