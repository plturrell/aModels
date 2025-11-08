package gemm

import (
	"os"
	"runtime"
	"strconv"
	"sync"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	cpu "golang.org/x/sys/cpu"
	"unsafe"
)

// Default block sizes for packed GEMM (can be overridden by env or heuristics).
const (
    kc = 256
    mc = 128
    nc = 128
)

// MatMulContiguous computes C = A * B for contiguous matrices.
// A is (m x k), B is (k x n), C returned as (m x n).
func MatMulContiguous(A, B *util.Matrix64) *util.Matrix64 {
    if A == nil || B == nil || A.Cols != B.Rows {
        return util.NewMatrix64(0, 0)
    }
    m, k, n := A.Rows, A.Cols, B.Cols
    C := util.NewMatrix64(m, n)
    if m == 0 || n == 0 || k == 0 {
        return C
    }

    // Outer blocking over N, K, M dimensions
    // Choose tile sizes (env overrides and simple heuristics)
    tmc, tnc, tkc := chooseTileSizes(m, n, k)
    // Packing mode for B: default column-major for better locality across K
    useColPack := true
    if v := os.Getenv("MATHS_B_PACK"); v != "" {
        if v == "row" { useColPack = false } else if v == "col" { useColPack = true }
    } else {
        // Heuristic: prefer col-pack on amd64 (uses native AVX2 col-pack kernel when built with avx2asm)
        // and row-pack on arm64 (uses NEON 8x4 kernel when built with neonasm).
        if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 { useColPack = true }
        if runtime.GOARCH == "arm64" && cpu.ARM64.HasASIMD { useColPack = false }
    }

    // Concurrency: parallelize across ic blocks; cap by threads
    maxThreads := runtime.GOMAXPROCS(0)
    if s := os.Getenv("MATHS_GEMM_THREADS"); s != "" {
        if v, err := strconv.Atoi(s); err == nil && v > 0 { maxThreads = v }
    }
    type task struct{ ic, im, jc, jn, pc, kn int }
    tasks := make(chan task, 8)
    var wg sync.WaitGroup

    worker := func() {
        defer wg.Done()
        // Per-worker packs to avoid contention
        aLocal := makeAlignedFloat64(tmc*tkc, 64)
        bLocal := makeAlignedFloat64(tkc*tnc, 64)
        var lastPC, lastJC, lastKN, lastJN = -1, -1, -1, -1
        for t := range tasks {
            // Re-pack B if panel changed
            if t.pc != lastPC || t.jc != lastJC || t.kn != lastKN || t.jn != lastJN {
                if useColPack {
                    packBColMajor(B, t.pc, t.kn, t.jc, t.jn, bLocal)
                } else {
                    packBRowMajor(B, t.pc, t.kn, t.jc, t.jn, bLocal)
                }
                lastPC, lastJC, lastKN, lastJN = t.pc, t.jc, t.kn, t.jn
            }
            packARowMajor(A, t.ic, t.im, t.pc, t.kn, aLocal)
            if useColPack {
                gemmBlockSelectColMajor(t.im, t.jn, t.kn, aLocal, bLocal, C, t.ic, t.jc)
            } else {
                gemmBlockSelect(t.im, t.jn, t.kn, aLocal, bLocal, C, t.ic, t.jc)
            }
        }
    }

    workers := maxThreads
    if workers < 1 { workers = 1 }
    blocks := 1
    if m > 0 { blocks = (m + tmc - 1) / tmc }
    if workers > blocks { workers = blocks }
    for i := 0; i < workers; i++ { wg.Add(1); go worker() }

    for jc := 0; jc < n; jc += tnc {
        jn := min(jc+tnc, n) - jc
        for pc := 0; pc < k; pc += tkc {
            kn := min(pc+tkc, k) - pc
            for ic := 0; ic < m; ic += tmc {
                im := min(ic+tmc, m) - ic
                tasks <- task{ic: ic, im: im, jc: jc, jn: jn, pc: pc, kn: kn}
            }
        }
    }
    close(tasks)
    wg.Wait()
    return C
}

// MatMul2D wrapper: accepts [][]float64, converts to contiguous, runs packed GEMM, returns [][]float64.
func MatMul2D(A, B [][]float64) [][]float64 {
    a := util.From2D(A)
    b := util.From2D(B)
    c := MatMulContiguous(a, b)
    return util.To2D(c)
}

// packARowMajor copies an (im x kn) panel from A starting at (ic,pc) into aPack row-major (row stride kn).
func packARowMajor(A *util.Matrix64, ic, im, pc, kn int, aPack []float64) {
    for i := 0; i < im; i++ {
        src := A.Data[(ic+i)*A.Stride+pc : (ic+i)*A.Stride+pc+kn]
        dst := aPack[i*kn : i*kn+kn]
        copy(dst, src)
    }
}

// packBRowMajor copies an (kn x jn) panel from B starting at (pc,jc) into bPack row-major (row stride jn).
func packBRowMajor(B *util.Matrix64, pc, kn, jc, jn int, bPack []float64) {
    for p := 0; p < kn; p++ {
        src := B.Data[(pc+p)*B.Stride+jc : (pc+p)*B.Stride+jc+jn]
        dst := bPack[p*jn : p*jn+jn]
        copy(dst, src)
    }
}

// gemmBlockRowMajor multiplies packed A(im x kn) with packed B(kn x jn) into C block at (ic,jc).
func gemmBlockRowMajor(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    // Simple triple loop over packed panels; compiler can auto-vectorize inner loop.
    // Further speedups can be achieved with small-register microkernels.
    for i := 0; i < im; i++ {
        cRow := C.Data[(ic+i)*C.Stride+jc : (ic+i)*C.Stride+jc+jn]
        aRow := aPack[i*kn : i*kn+kn]
        // Unroll j in small steps for better ILP on common sizes
        j := 0
        for ; j+3 < jn; j += 4 {
            s0, s1, s2, s3 := 0.0, 0.0, 0.0, 0.0
            for p := 0; p < kn; p++ {
                ap := aRow[p]
                bp := bPack[p*jn+j : p*jn+j+4]
                s0 += ap * bp[0]
                s1 += ap * bp[1]
                s2 += ap * bp[2]
                s3 += ap * bp[3]
            }
            cRow[j+0] += s0
            cRow[j+1] += s1
            cRow[j+2] += s2
            cRow[j+3] += s3
        }
        for ; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ {
                s += aRow[p] * bPack[p*jn+j]
            }
            cRow[j] += s
        }
    }
}

// bestGemmImpl returns the chosen implementation identifier for diagnostics.
func bestGemmImpl() string {
    // Runtime dispatch summary string.
    if v := os.Getenv("MATHS_FORCE_KERNEL"); v != "" { return v }
    bpack := os.Getenv("MATHS_B_PACK")
    if bpack == "" { bpack = "col" }
    if runtime.GOARCH == "amd64" {
        if cpu.X86.HasAVX512F { return "packed-go-par(avx512-8x4,bpack="+bpack+")" }
        if cpu.X86.HasAVX2 { return "packed-go-par(avx2-8x4,bpack="+bpack+")" }
        return "packed-go-par(scalar,bpack="+bpack+")"
    }
    if runtime.GOARCH == "arm64" {
        if cpu.ARM64.HasASIMD { return "packed-go-par(neon-8x4,bpack="+bpack+")" }
        return "packed-go-par(scalar,bpack="+bpack+")"
    }
    return "packed-go-par"
}

// chooseTileSizes selects (mc,nc,kc) based on env overrides and simple heuristics.
func chooseTileSizes(m, n, k int) (int, int, int) {
    // Base defaults
    tmc, tnc, tkc := mc, nc, kc
    // Env overrides
    if s := os.Getenv("MATHS_GEMM_MC"); s != "" { if v, err := strconv.Atoi(s); err == nil && v > 0 { tmc = v } }
    if s := os.Getenv("MATHS_GEMM_NC"); s != "" { if v, err := strconv.Atoi(s); err == nil && v > 0 { tnc = v } }
    if s := os.Getenv("MATHS_GEMM_KC"); s != "" { if v, err := strconv.Atoi(s); err == nil && v > 0 { tkc = v } }

    // Heuristics: ensure multiples of microkernel tile sizes and adapt to problem size
    if runtime.GOARCH == "amd64" {
        // Prefer columns multiple of 16 (two 8x8 halves) and rows multiple of 8
        if tmc%8 != 0 { tmc = (tmc/8)*8 }
        if tnc%8 != 0 { tnc = (tnc/8)*8 }
        if cpu.X86.HasAVX2 || cpu.X86.HasAVX512F {
            if tnc < 128 && n >= 1024 { tnc = 128 }
            if tkc < 256 && k >= 1024 { tkc = 256 }
        }
    } else if runtime.GOARCH == "arm64" {
        if tmc%8 != 0 { tmc = (tmc/8)*8 }
        if tnc%4 != 0 { tnc = (tnc/4)*4 }
        if tkc < 192 && k >= 1024 { tkc = 192 }
    }
    if tmc <= 0 { tmc = mc }
    if tnc <= 0 { tnc = nc }
    if tkc <= 0 { tkc = kc }
    return tmc, tnc, tkc
}

// gemmBlockSelect chooses a block kernel based on runtime CPU features and tile sizes.
func gemmBlockSelect(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    // Allow forcing generic path for benchmarking
    if os.Getenv("MATHS_DISABLE_MICROKERNEL") == "1" {
        gemmBlockRowMajor(im, jn, kn, aPack, bPack, C, ic, jc)
        return
    }
    use8x4 := false
    if runtime.GOARCH == "amd64" {
        use8x4 = cpu.X86.HasAVX2 || cpu.X86.HasAVX512F
    } else if runtime.GOARCH == "arm64" {
        use8x4 = cpu.ARM64.HasASIMD
    }
    if use8x4 && im >= 8 && jn >= 16 {
        if asmGemm8x16(aPack, bPack, C, ic, jc, im, jn, kn) { return }
        // Fall back to 8x8 tiles twice
        gemmBlockRowMajor8x8(im, jn, kn, aPack, bPack, C, ic, jc)
        return
    }
    if use8x4 && im >= 8 && jn >= 8 {
        if asmGemm8x8(aPack, bPack, C, ic, jc, im, jn, kn) { return }
        gemmBlockRowMajor8x8(im, jn, kn, aPack, bPack, C, ic, jc)
        return
    }
    if use8x4 && im >= 8 && jn >= 4 {
        if asmGemm8x4(aPack, bPack, C, ic, jc, im, jn, kn) { return }
        gemmBlockRowMajor8x4(im, jn, kn, aPack, bPack, C, ic, jc)
        return
    }
    gemmBlockRowMajor(im, jn, kn, aPack, bPack, C, ic, jc)
}

// Column-major B packed microkernel selection.
func gemmBlockSelectColMajor(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    // For column-packed B, try asm via temporary row repack per tile when available.
    if tryAsmColPack(im, jn, kn, aPack, bPack, C, ic, jc) { return }
    // Prefer 8x8 microkernel when possible, else 8x4, else generic.
    if im >= 8 && jn >= 8 {
        gemmBlockColMajor8x8(im, jn, kn, aPack, bPack, C, ic, jc)
        return
    }
    if im >= 8 && jn >= 4 {
        gemmBlockColMajor8x4(im, jn, kn, aPack, bPack, C, ic, jc)
        return
    }
    gemmBlockColMajor(im, jn, kn, aPack, bPack, C, ic, jc)
}

// Column-major B pack: B panel [pc:pc+kn, jc:jc+jn] stored as jn columns each of length kn contiguous.
func packBColMajor(B *util.Matrix64, pc, kn, jc, jn int, bPack []float64) {
    for j := 0; j < jn; j++ {
        dst := bPack[j*kn : j*kn+kn]
        for p := 0; p < kn; p++ {
            dst[p] = B.Data[(pc+p)*B.Stride + (jc+j)]
        }
    }
}

// Generic column-major microkernel for any sizes.
func gemmBlockColMajor(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    for i := 0; i < im; i++ {
        cRow := C.Data[(ic+i)*C.Stride+jc : (ic+i)*C.Stride+jc+jn]
        aRow := aPack[i*kn : i*kn+kn]
        // Unroll columns in 4-wide chunks
        j := 0
        for ; j+3 < jn; j += 4 {
            s0, s1, s2, s3 := 0.0, 0.0, 0.0, 0.0
            b0 := bPack[(j+0)*kn : (j+0)*kn+kn]
            b1 := bPack[(j+1)*kn : (j+1)*kn+kn]
            b2 := bPack[(j+2)*kn : (j+2)*kn+kn]
            b3 := bPack[(j+3)*kn : (j+3)*kn+kn]
            for p := 0; p < kn; p++ {
                ap := aRow[p]
                s0 += ap * b0[p]
                s1 += ap * b1[p]
                s2 += ap * b2[p]
                s3 += ap * b3[p]
            }
            cRow[j+0] += s0
            cRow[j+1] += s1
            cRow[j+2] += s2
            cRow[j+3] += s3
        }
        for ; j < jn; j++ {
            s := 0.0
            bj := bPack[j*kn : j*kn+kn]
            for p := 0; p < kn; p++ { s += aRow[p] * bj[p] }
            cRow[j] += s
        }
    }
}

// 8x4 microkernel for column-packed B.
func gemmBlockColMajor8x4(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    i := 0
    for ; i+7 < im; i += 8 {
        j := 0
        for ; j+3 < jn; j += 4 {
            var s [8][4]float64
            b0 := bPack[(j+0)*kn : (j+0)*kn+kn]
            b1 := bPack[(j+1)*kn : (j+1)*kn+kn]
            b2 := bPack[(j+2)*kn : (j+2)*kn+kn]
            b3 := bPack[(j+3)*kn : (j+3)*kn+kn]
            for p := 0; p < kn; p++ {
                bv0, bv1, bv2, bv3 := b0[p], b1[p], b2[p], b3[p]
                base := (i*kn + p)
                a0 := aPack[base+0*kn]
                a1 := aPack[base+1*kn]
                a2 := aPack[base+2*kn]
                a3 := aPack[base+3*kn]
                a4 := aPack[base+4*kn]
                a5 := aPack[base+5*kn]
                a6 := aPack[base+6*kn]
                a7 := aPack[base+7*kn]
                s[0][0] += a0 * bv0; s[0][1] += a0 * bv1; s[0][2] += a0 * bv2; s[0][3] += a0 * bv3
                s[1][0] += a1 * bv0; s[1][1] += a1 * bv1; s[1][2] += a1 * bv2; s[1][3] += a1 * bv3
                s[2][0] += a2 * bv0; s[2][1] += a2 * bv1; s[2][2] += a2 * bv2; s[2][3] += a2 * bv3
                s[3][0] += a3 * bv0; s[3][1] += a3 * bv1; s[3][2] += a3 * bv2; s[3][3] += a3 * bv3
                s[4][0] += a4 * bv0; s[4][1] += a4 * bv1; s[4][2] += a4 * bv2; s[4][3] += a4 * bv3
                s[5][0] += a5 * bv0; s[5][1] += a5 * bv1; s[5][2] += a5 * bv2; s[5][3] += a5 * bv3
                s[6][0] += a6 * bv0; s[6][1] += a6 * bv1; s[6][2] += a6 * bv2; s[6][3] += a6 * bv3
                s[7][0] += a7 * bv0; s[7][1] += a7 * bv1; s[7][2] += a7 * bv2; s[7][3] += a7 * bv3
            }
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride + jc + j : (ic+i+r)*C.Stride + jc + j + 4]
                cRow[0] += s[r][0]
                cRow[1] += s[r][1]
                cRow[2] += s[r][2]
                cRow[3] += s[r][3]
            }
        }
        if j < jn {
            // tail columns via generic path for these 8 rows
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride + jc + j : (ic+i+r)*C.Stride + jc + jn]
                aRow := aPack[(i+r)*kn : (i+r)*kn+kn]
                for jj := j; jj < jn; jj++ {
                    s := 0.0
                    bj := bPack[jj*kn : jj*kn+kn]
                    for p := 0; p < kn; p++ { s += aRow[p] * bj[p] }
                    cRow[jj-j] += s
                }
            }
        }
    }
    if i < im {
        gemmBlockColMajor(im-i, jn, kn, aPack[i*kn:], bPack, C, ic+i, jc)
    }
}

// 8x8 microkernel for column-packed B.
func gemmBlockColMajor8x8(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    i := 0
    for ; i+7 < im; i += 8 {
        j := 0
        for ; j+7 < jn; j += 8 {
            var s [8][8]float64
            bcols := [8][]float64{}
            for c := 0; c < 8; c++ { bcols[c] = bPack[(j+c)*kn : (j+c)*kn+kn] }
            for p := 0; p < kn; p++ {
                base := (i*kn + p)
                a0 := aPack[base+0*kn]
                a1 := aPack[base+1*kn]
                a2 := aPack[base+2*kn]
                a3 := aPack[base+3*kn]
                a4 := aPack[base+4*kn]
                a5 := aPack[base+5*kn]
                a6 := aPack[base+6*kn]
                a7 := aPack[base+7*kn]
                for c := 0; c < 8; c++ {
                    bv := bcols[c][p]
                    s[0][c] += a0 * bv
                    s[1][c] += a1 * bv
                    s[2][c] += a2 * bv
                    s[3][c] += a3 * bv
                    s[4][c] += a4 * bv
                    s[5][c] += a5 * bv
                    s[6][c] += a6 * bv
                    s[7][c] += a7 * bv
                }
            }
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride + jc + j : (ic+i+r)*C.Stride + jc + j + 8]
                for c := 0; c < 8; c++ { cRow[c] += s[r][c] }
            }
        }
        if j < jn {
            // tail columns via generic path for these 8 rows
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride + jc + j : (ic+i+r)*C.Stride + jc + jn]
                aRow := aPack[(i+r)*kn : (i+r)*kn+kn]
                for jj := j; jj < jn; jj++ {
                    s := 0.0
                    bj := bPack[jj*kn : jj*kn+kn]
                    for p := 0; p < kn; p++ { s += aRow[p] * bj[p] }
                    cRow[jj-j] += s
                }
            }
        }
    }
    if i < im {
        gemmBlockColMajor(im-i, jn, kn, aPack[i*kn:], bPack, C, ic+i, jc)
    }
}

// makeAlignedFloat64 returns a slice of length n such that the first element's
// address is aligned to 'align' bytes (align should be a power of two).
func makeAlignedFloat64(n int, align int) []float64 {
    if n <= 0 { return nil }
    // Overallocate to ensure we can find an aligned start within the buffer.
    extra := (align + 7) / 8 // number of float64 slots to cover 'align' bytes
    buf := make([]float64, n+extra)
    base := uintptr(unsafe.Pointer(&buf[0]))
    mask := uintptr(align - 1)
    // Compute how many bytes to skip to reach alignment boundary.
    if (base & mask) == 0 {
        return buf[:n]
    }
    skipBytes := (align - int(base&mask)) & (align - 1)
    skip := skipBytes / 8
    return buf[skip : skip+n]
}

// gemmBlockRowMajor8x4 computes C block using an 8x4 microkernel on top of packed A(im x kn) and B(kn x jn).
func gemmBlockRowMajor8x4(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    // Process 8 rows x 4 cols tiles
    i := 0
    for ; i+7 < im; i += 8 {
        j := 0
        for ; j+3 < jn; j += 4 {
            var s [8][4]float64
            // Accumulate over K
            for p := 0; p < kn; p++ {
                // Load B row (4 contiguous elements)
                b0 := bPack[p*jn+j+0]
                b1 := bPack[p*jn+j+1]
                b2 := bPack[p*jn+j+2]
                b3 := bPack[p*jn+j+3]
                // FMA across 8 A rows
                base := (i*kn + p) // row-major aPack stride kn
                a0 := aPack[base+0*kn]
                a1 := aPack[base+1*kn]
                a2 := aPack[base+2*kn]
                a3 := aPack[base+3*kn]
                a4 := aPack[base+4*kn]
                a5 := aPack[base+5*kn]
                a6 := aPack[base+6*kn]
                a7 := aPack[base+7*kn]
                s[0][0] += a0 * b0; s[0][1] += a0 * b1; s[0][2] += a0 * b2; s[0][3] += a0 * b3
                s[1][0] += a1 * b0; s[1][1] += a1 * b1; s[1][2] += a1 * b2; s[1][3] += a1 * b3
                s[2][0] += a2 * b0; s[2][1] += a2 * b1; s[2][2] += a2 * b2; s[2][3] += a2 * b3
                s[3][0] += a3 * b0; s[3][1] += a3 * b1; s[3][2] += a3 * b2; s[3][3] += a3 * b3
                s[4][0] += a4 * b0; s[4][1] += a4 * b1; s[4][2] += a4 * b2; s[4][3] += a4 * b3
                s[5][0] += a5 * b0; s[5][1] += a5 * b1; s[5][2] += a5 * b2; s[5][3] += a5 * b3
                s[6][0] += a6 * b0; s[6][1] += a6 * b1; s[6][2] += a6 * b2; s[6][3] += a6 * b3
                s[7][0] += a7 * b0; s[7][1] += a7 * b1; s[7][2] += a7 * b2; s[7][3] += a7 * b3
            }
            // Store results
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride+jc+j : (ic+i+r)*C.Stride+jc+j+4]
                cRow[0] += s[r][0]
                cRow[1] += s[r][1]
                cRow[2] += s[r][2]
                cRow[3] += s[r][3]
            }
        }
        // Remainder columns for these 8 rows
        if j < jn {
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride+jc+j : (ic+i+r)*C.Stride+jc+jn]
                aRow := aPack[(i+r)*kn : (i+r)*kn+kn]
                for jj := j; jj < jn; jj++ {
                    s := 0.0
                    for p := 0; p < kn; p++ { s += aRow[p] * bPack[p*jn+jj] }
                    cRow[jj-j] += s
                }
            }
        }
    }
    // Remaining rows
    if i < im {
        gemmBlockRowMajor(im-i, jn, kn, aPack[i*kn:], bPack, C, ic+i, jc)
    }
}

// gemmBlockRowMajor8x8 computes C block using an 8x8 microkernel on top of packed A(im x kn) and B(kn x jn).
func gemmBlockRowMajor8x8(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) {
    i := 0
    for ; i+7 < im; i += 8 {
        j := 0
        for ; j+7 < jn; j += 8 {
            var s [8][8]float64
            for p := 0; p < kn; p++ {
                // Load 8 contiguous B values for current p row
                bp := bPack[p*jn+j : p*jn+j+8]
                base := (i*kn + p)
                a0 := aPack[base+0*kn]
                a1 := aPack[base+1*kn]
                a2 := aPack[base+2*kn]
                a3 := aPack[base+3*kn]
                a4 := aPack[base+4*kn]
                a5 := aPack[base+5*kn]
                a6 := aPack[base+6*kn]
                a7 := aPack[base+7*kn]
                // Unrolled FMA for 8 cols
                for c := 0; c < 8; c++ {
                    bv := bp[c]
                    s[0][c] += a0 * bv
                    s[1][c] += a1 * bv
                    s[2][c] += a2 * bv
                    s[3][c] += a3 * bv
                    s[4][c] += a4 * bv
                    s[5][c] += a5 * bv
                    s[6][c] += a6 * bv
                    s[7][c] += a7 * bv
                }
            }
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride+jc+j : (ic+i+r)*C.Stride+jc+j+8]
                for c := 0; c < 8; c++ { cRow[c] += s[r][c] }
            }
        }
        // Remainder columns for these 8 rows
        if j < jn {
            for r := 0; r < 8; r++ {
                cRow := C.Data[(ic+i+r)*C.Stride+jc+j : (ic+i+r)*C.Stride+jc+jn]
                aRow := aPack[(i+r)*kn : (i+r)*kn+kn]
                for jj := j; jj < jn; jj++ {
                    s := 0.0
                    for p := 0; p < kn; p++ { s += aRow[p] * bPack[p*jn+jj] }
                    cRow[jj-j] += s
                }
            }
        }
    }
    if i < im {
        gemmBlockRowMajor(im-i, jn, kn, aPack[i*kn:], bPack, C, ic+i, jc)
    }
}
