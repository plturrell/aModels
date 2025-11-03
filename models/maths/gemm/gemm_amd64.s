//go:build amd64

#include "textflag.h"

// func gemm8x4_kernel(a *float64, b *float64, kn int, jn int, c *float64, cstride int)
TEXT ·gemm8x4_kernel(SB), NOSPLIT, $0-48
    // Load args
    MOVQ a+0(FP), R8      // a base
    MOVQ b+8(FP), R9      // b base
    MOVQ kn+16(FP), R10   // kn
    MOVQ jn+24(FP), R11   // jn
    MOVQ c+32(FP), R12    // c base
    MOVQ cstride+40(FP), R13 // c stride

    XORQ R14, R14         // p = 0
loop_p:
    CMPQ R14, R10
    JGE done
    // Compute b row pointer: br = b + (p*jn)*8
    MOVQ R14, AX
    IMULQ R11, AX
    SHLQ $3, AX
    LEAQ (R9)(AX*1), R15
    // Load b0..b3
    MOVSD 0(R15), X1
    MOVSD 8(R15), X2
    MOVSD 16(R15), X3
    MOVSD 24(R15), X4

    XORQ CX, CX           // r = 0
loop_r:
    CMPQ CX, $8
    JGE next_p
    // a index: off = (r*kn + p)*8
    MOVQ CX, DX
    IMULQ R10, DX
    ADDQ R14, DX
    SHLQ $3, DX
    MOVSD (R8)(DX*1), X0    // a

    // c row pointer: cptr = c + (r*cstride)*8
    MOVQ CX, BX
    IMULQ R13, BX
    SHLQ $3, BX
    LEAQ (R12)(BX*1), DI

    // c[0] += a*b0
    MOVSD 0(DI), X5
    MOVSD X0, X6
    MULSD X1, X6
    ADDSD X6, X5
    MOVSD X5, 0(DI)
    // c[1] += a*b1
    MOVSD 8(DI), X5
    MOVSD X0, X6
    MULSD X2, X6
    ADDSD X6, X5
    MOVSD X5, 8(DI)
    // c[2] += a*b2
    MOVSD 16(DI), X5
    MOVSD X0, X6
    MULSD X3, X6
    ADDSD X6, X5
    MOVSD X5, 16(DI)
    // c[3] += a*b3
    MOVSD 24(DI), X5
    MOVSD X0, X6
    MULSD X4, X6
    ADDSD X6, X5
    MOVSD X5, 24(DI)

    INCQ CX
    JMP loop_r

next_p:
    INCQ R14
    JMP loop_p

done:
    RET

