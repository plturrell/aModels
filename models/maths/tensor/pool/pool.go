package pool

import (
    "fmt"
    "sync"
)

// BufferPool manages reusable 1D buffers
type BufferPool struct {
    pool sync.Pool
}

// NewBufferPool creates a buffer pool
func NewBufferPool() *BufferPool {
    return &BufferPool{
        pool: sync.Pool{ New: func() interface{} { return make([]float64, 0, 1024) } },
    }
}

// Get retrieves a buffer from pool
func (bp *BufferPool) Get(size int) []float64 {
    buf := bp.pool.Get().([]float64)
    if cap(buf) < size { buf = make([]float64, size) }
    return buf[:size]
}

// Put returns a buffer to pool
func (bp *BufferPool) Put(buf []float64) { bp.pool.Put(buf[:0]) }

// MatrixBufferPool manages reusable 2D buffers
type MatrixBufferPool struct { pool sync.Pool }

// NewMatrixBufferPool creates a matrix buffer pool
func NewMatrixBufferPool() *MatrixBufferPool {
    return &MatrixBufferPool{
        pool: sync.Pool{ New: func() interface{} { return make([][]float64, 0) } },
    }
}

// Get retrieves a matrix buffer
func (mbp *MatrixBufferPool) Get(m, n int) [][]float64 {
    mat := mbp.pool.Get().([][]float64)
    if cap(mat) < m { mat = make([][]float64, m) }
    mat = mat[:m]
    for i := 0; i < m; i++ {
        if cap(mat[i]) < n { mat[i] = make([]float64, n) }
        mat[i] = mat[i][:n]
    }
    return mat
}

// Put returns a matrix buffer to pool
func (mbp *MatrixBufferPool) Put(mat [][]float64) { mbp.pool.Put(mat[:0]) }

// MatrixPool manages matrices in size-keyed pools
type MatrixPool struct {
    pools map[string]*sync.Pool
    mu    sync.RWMutex
}

// NewMatrixPool creates a matrix pool
func NewMatrixPool() *MatrixPool { return &MatrixPool{ pools: make(map[string]*sync.Pool) } }

// Get retrieves a matrix from pool
func (mp *MatrixPool) Get(m, n int) [][]float64 {
    key := fmt.Sprintf("%d_%d", m, n)
    mp.mu.RLock()
    pool, exists := mp.pools[key]
    mp.mu.RUnlock()
    if !exists {
        mp.mu.Lock()
        pool = &sync.Pool{ New: func() interface{} {
            mat := make([][]float64, m)
            for i := 0; i < m; i++ { mat[i] = make([]float64, n) }
            return mat
        } }
        mp.pools[key] = pool
        mp.mu.Unlock()
    }
    return pool.Get().([][]float64)
}

// Put returns a matrix to pool
func (mp *MatrixPool) Put(mat [][]float64) {
    if len(mat) == 0 { return }
    m, n := len(mat), len(mat[0])
    key := fmt.Sprintf("%d_%d", m, n)
    mp.mu.RLock()
    pool, exists := mp.pools[key]
    mp.mu.RUnlock()
    if exists {
        for i := 0; i < m; i++ { for j := 0; j < n; j++ { mat[i][j] = 0 } }
        pool.Put(mat)
    }
}
