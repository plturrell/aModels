package linalg

import (
    "fmt"
    "os"
    "runtime"
    "strconv"
    "strings"
    "sync"
)

// BLASBackend represents available BLAS implementations
type BLASBackend int

const (
    BLASNone BLASBackend = iota
    BLASAccelerate // macOS Accelerate framework
    BLASOpenBLAS   // OpenBLAS
    BLASMKL        // Intel MKL
)

// BLASConfig holds BLAS configuration
type BLASConfig struct {
    Backend    BLASBackend
    NumThreads int
    Available  bool
}

var (
    blasConfig     BLASConfig
    blasConfigOnce sync.Once
)

// InitBLAS initializes BLAS backend with thread control
func InitBLAS(numThreads int) *BLASConfig {
    blasConfigOnce.Do(func() {
        blasConfig = detectBLAS()
        if numThreads <= 0 { numThreads = runtime.NumCPU() }
        blasConfig.NumThreads = numThreads
        configureBLASThreads(numThreads)
    })
    return &blasConfig
}

// GetBLASInfo returns current BLAS configuration
func GetBLASInfo() string {
    config := InitBLAS(0)
    backendName := "Pure Go (no BLAS)"
    switch config.Backend {
    case BLASAccelerate:
        backendName = "macOS Accelerate"
    case BLASOpenBLAS:
        backendName = "OpenBLAS"
    case BLASMKL:
        backendName = "Intel MKL"
    }
    return fmt.Sprintf("Backend: %s, Threads: %d, Available: %v", backendName, config.NumThreads, config.Available)
}

// ---- internal helpers ----

func detectBLAS() BLASConfig {
    if runtime.GOOS == "darwin" { return BLASConfig{ Backend: BLASAccelerate, Available: true } }
    if os.Getenv("OPENBLAS_NUM_THREADS") != "" || checkLibrary("libopenblas") { return BLASConfig{ Backend: BLASOpenBLAS, Available: true } }
    if os.Getenv("MKL_NUM_THREADS") != "" || checkLibrary("libmkl_rt") { return BLASConfig{ Backend: BLASMKL, Available: true } }
    return BLASConfig{ Backend: BLASNone, Available: false }
}

func configureBLASThreads(numThreads int) {
    threadsStr := strconv.Itoa(numThreads)
    threadVars := []string{
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    for _, varName := range threadVars { os.Setenv(varName, threadsStr) }
}

func checkLibrary(_ string) bool { return false }

// PreferredEigenImpl inspects env to select implementation
func PreferredEigenImpl() string { return strings.ToLower(strings.TrimSpace(os.Getenv("MATHS_EIGEN_PREFERRED"))) }

