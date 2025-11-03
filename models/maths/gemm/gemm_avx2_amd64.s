//go:build amd64 && avx2asm

#include "textflag.h"

// func gemm8x4_kernel_avx2(a *float64, b *float64, kn int, jn int, c *float64, cstride int)
TEXT ·gemm8x4_kernel_avx2(SB), NOSPLIT, $8-48
    // Load args
    MOVQ a+0(FP), R8      // a base
    MOVQ b+8(FP), R9      // b base
    MOVQ kn+16(FP), R10   // kn
    MOVQ jn+24(FP), R11   // jn
    MOVQ c+32(FP), R12    // c base
    MOVQ cstride+40(FP), R13 // c stride (elements)

    // Save kn (loop bound) on stack
    MOVQ R10, -8(SP)

    // Compute row stride bytes for A: kn * 8
    MOVQ R10, AX
    SHLQ $3, AX            // AX = kn*8

    // Compute A row base pointers for 8 rows
    MOVQ R8, DI            // row0 = a
    LEAQ (DI)(AX*1), SI      // row1 = row0 + kn*8
    LEAQ (SI)(AX*1), BX      // row2
    LEAQ (BX)(AX*1), CX      // row3
    LEAQ (CX)(AX*1), DX      // row4
    LEAQ (DX)(AX*1), BP      // row5
    LEAQ (BP)(AX*1), R8      // row6 reuse R8
    LEAQ (R8)(AX*1), R10     // row7 reuse R10

    // Init accumulators Y0..Y7 to zero
    VXORPD Y0, Y0, Y0
    VXORPD Y1, Y1, Y1
    VXORPD Y2, Y2, Y2
    VXORPD Y3, Y3, Y3
    VXORPD Y4, Y4, Y4
    VXORPD Y5, Y5, Y5
    VXORPD Y6, Y6, Y6
    VXORPD Y7, Y7, Y7

    // p loop
    XORQ R14, R14          // p = 0
    MOVQ R9, R15           // B row pointer = b

loop_p:
    CMPQ R14, -8(SP)
    JGE done_acc

    VMOVUPD (R15), Y8          // load 4 doubles from B row p

    // Broadcast a[r,p] and FMA into accumulators
    VBROADCASTSD (DI)(R14*8), Y9
    VADDPD Y0, Y0, Y0
    VFMADD231PD Y8, Y9, Y0

    VBROADCASTSD (SI)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y1

    VBROADCASTSD (BX)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y2

    VBROADCASTSD (CX)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y3

    VBROADCASTSD (DX)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y4

    VBROADCASTSD (BP)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y5

    VBROADCASTSD (R8)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y6

    VBROADCASTSD (R10)(R14*8), Y9
    VFMADD231PD Y8, Y9, Y7

    // Next p
    INCQ R14
    LEAQ (R15)(R11*8), R15
    JMP loop_p

done_acc:
    // Store C rows (8 rows x 4 cols)
    MOVQ R13, AX
    SHLQ $3, AX            // bytes per row
    MOVQ R12, DI           // row0
    LEAQ (DI)(AX*1), SI    // row1
    LEAQ (SI)(AX*1), BX    // row2
    LEAQ (BX)(AX*1), CX    // row3
    LEAQ (CX)(AX*1), DX    // row4
    LEAQ (DX)(AX*1), BP    // row5
    LEAQ (BP)(AX*1), R9    // row6
    LEAQ (R9)(AX*1), R11   // row7

    VMOVUPD (DI), Y10
    VADDPD Y0, Y10, Y10
    VMOVUPD Y10, (DI)

    VMOVUPD (SI), Y10
    VADDPD Y1, Y10, Y10
    VMOVUPD Y10, (SI)

    VMOVUPD (BX), Y10
    VADDPD Y2, Y10, Y10
    VMOVUPD Y10, (BX)

    VMOVUPD (CX), Y10
    VADDPD Y3, Y10, Y10
    VMOVUPD Y10, (CX)

    VMOVUPD (DX), Y10
    VADDPD Y4, Y10, Y10
    VMOVUPD Y10, (DX)

    VMOVUPD (BP), Y10
    VADDPD Y5, Y10, Y10
    VMOVUPD Y10, (BP)

    VMOVUPD (R9), Y10
    VADDPD Y6, Y10, Y10
    VMOVUPD Y10, (R9)

    VMOVUPD (R11), Y10
    VADDPD Y7, Y10, Y10
    VMOVUPD Y10, (R11)

    RET

