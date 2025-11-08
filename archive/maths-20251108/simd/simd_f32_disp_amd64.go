//go:build amd64 && !simd_asm_avx512
// +build amd64,!simd_asm_avx512

package simd

import (
    cpu "golang.org/x/sys/cpu"
    lanes "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/lanes"
)

// SIMDFusedMultiplyAddF32 returns a*b + c element-wise for float32 on amd64.
// Prefers AVX2 8-wide unrolling when available; otherwise falls back to scalar.
func SIMDFusedMultiplyAddF32(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("infrastructure/maths/simd.SIMDFusedMultiplyAddF32: length mismatch") }
    if cpu.X86.HasAVX2 {
        return lanes.LaneFMAF32_AVX2(a, b, c)
    }
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}

