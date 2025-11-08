//go:build amd64 && avx2asm

#include "textflag.h"

// Thin wrapper: route FMA entry to the AVX2 kernel implementation.
// func gemm8x4_kernel_avx2_fma(a *float64, b *float64, kn int, jn int, c *float64, cstride int)
TEXT ·gemm8x4_kernel_avx2_fma(SB), NOSPLIT, $0-48
    CALL ·gemm8x4_kernel_avx2(SB)
    RET
