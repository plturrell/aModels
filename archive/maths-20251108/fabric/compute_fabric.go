package fabric

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sync"
	"time"
)

// ComputeRequest represents a computation request in the fabric
type ComputeRequest struct {
	ID        string
	Operation string
	Args      []interface{}
	Result    chan interface{}
	Error     chan error
	Priority  int
	AgentID   string
	Timestamp time.Time
}

// ComputeResult represents the result of a computation
type ComputeResult struct {
	Value     interface{}
	Duration  time.Duration
	Algorithm string
	CacheHit  bool
}

// ComputeFabric provides distributed computation across agents
type ComputeFabric struct {
	mu             sync.RWMutex
	pending        map[string]*ComputeRequest
	completed      map[string]*ComputeResult
	workers        int
	requestQueue   chan *ComputeRequest
	agentLoads     map[string]int
	algorithmStats map[string]map[string]float64
	shutdown       chan struct{}
	wg             sync.WaitGroup
}

// NewComputeFabric creates a new compute fabric
func NewComputeFabric(workers int) *ComputeFabric {
	cf := &ComputeFabric{
		pending:        make(map[string]*ComputeRequest),
		completed:      make(map[string]*ComputeResult),
		workers:        workers,
		requestQueue:   make(chan *ComputeRequest, 1000),
		agentLoads:     make(map[string]int),
		algorithmStats: make(map[string]map[string]float64),
		shutdown:       make(chan struct{}),
	}

	// Start worker pool
	for i := 0; i < workers; i++ {
		cf.wg.Add(1)
		go cf.worker(i)
	}

	return cf
}

// Submit submits a computation request to the fabric
func (cf *ComputeFabric) Submit(req *ComputeRequest) error {
	cf.mu.Lock()

	// Check if already computed
	if result, ok := cf.completed[req.ID]; ok {
		cf.mu.Unlock()
		req.Result <- result
		return nil
	}

	// Check if already pending
	if existing, ok := cf.pending[req.ID]; ok {
		cf.mu.Unlock()
		// Wait for existing computation
		select {
		case result := <-existing.Result:
			req.Result <- result
		case err := <-existing.Error:
			req.Error <- err
		}
		return nil
	}

	// Set timestamp and add to pending
	req.Timestamp = time.Now()
	cf.pending[req.ID] = req
	cf.mu.Unlock()

	// Submit to queue
	select {
	case cf.requestQueue <- req:
		return nil
	default:
		cf.mu.Lock()
		delete(cf.pending, req.ID)
		cf.mu.Unlock()
		return fmt.Errorf("compute fabric queue full")
	}
}

// worker processes computation requests
func (cf *ComputeFabric) worker(id int) {
	defer cf.wg.Done()

	for {
		select {
		case req := <-cf.requestQueue:
			cf.processRequest(req)
		case <-cf.shutdown:
			return
		}
	}
}

// processRequest processes a single computation request
func (cf *ComputeFabric) processRequest(req *ComputeRequest) {
	startTime := time.Now()

	// Execute computation
	result, err := cf.execute(req)
	duration := time.Since(startTime)

	cf.mu.Lock()

	// Remove from pending
	delete(cf.pending, req.ID)

	// Store result
	if err == nil {
		cf.completed[req.ID] = &ComputeResult{
			Value:     result,
			Duration:  duration,
			Algorithm: cf.selectAlgorithm(req),
			CacheHit:  false,
		}
	}

	// Update agent load
	cf.agentLoads[req.AgentID]--
	if cf.agentLoads[req.AgentID] < 0 {
		cf.agentLoads[req.AgentID] = 0
	}

	cf.mu.Unlock()

	// Send result
	if err != nil {
		req.Error <- err
	} else {
		req.Result <- result
	}
}

// execute performs the actual computation
func (cf *ComputeFabric) execute(req *ComputeRequest) (interface{}, error) {
	switch req.Operation {
	case "cosine":
		return cf.executeCosine(req.Args)
	case "dot":
		return cf.executeDot(req.Args)
	case "matmul":
		return cf.executeMatMul(req.Args)
	case "normalize":
		return cf.executeNormalize(req.Args)
	default:
		return nil, fmt.Errorf("unknown operation: %s", req.Operation)
	}
}

// executeCosine computes cosine similarity
func (cf *ComputeFabric) executeCosine(args []interface{}) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("cosine requires 2 arguments")
	}

	v1, ok1 := args[0].([]float64)
	v2, ok2 := args[1].([]float64)

	if !ok1 || !ok2 {
		return nil, fmt.Errorf("cosine requires []float64 arguments")
	}

	_ = v1
	_ = v2

	// This would call the actual math function
	// result := maths.CosineAuto(v1, v2)
	result := 0.0 // Placeholder

	return result, nil
}

// executeDot computes dot product
func (cf *ComputeFabric) executeDot(args []interface{}) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("dot requires 2 arguments")
	}

	v1, ok1 := args[0].([]float64)
	v2, ok2 := args[1].([]float64)

	if !ok1 || !ok2 {
		return nil, fmt.Errorf("dot requires []float64 arguments")
	}

	_ = v1
	_ = v2

	// This would call the actual math function
	// result, _ := maths.DotAuto(v1, v2)
	result := 0.0 // Placeholder

	return result, nil
}

// executeMatMul computes matrix multiplication
func (cf *ComputeFabric) executeMatMul(args []interface{}) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("matmul requires 2 arguments")
	}

	a, ok1 := args[0].([][]float64)
	b, ok2 := args[1].([][]float64)

	if !ok1 || !ok2 {
		return nil, fmt.Errorf("matmul requires [][]float64 arguments")
	}

	_ = b

	// This would call the actual math function
	// result := maths.MatMul(a, b)
	result := make([][]float64, len(a)) // Placeholder

	return result, nil
}

// executeNormalize normalizes a vector
func (cf *ComputeFabric) executeNormalize(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("normalize requires 1 argument")
	}

	v, ok := args[0].([]float64)
	if !ok {
		return nil, fmt.Errorf("normalize requires []float64 argument")
	}

	// This would call the actual math function
	// result := maths.NormalizeVector(v)
	result := make([]float64, len(v)) // Placeholder

	return result, nil
}

// selectAlgorithm selects the best algorithm for a request
func (cf *ComputeFabric) selectAlgorithm(req *ComputeRequest) string {
	// Simple heuristic based on operation and size
	switch req.Operation {
	case "cosine", "dot":
		if cf.getVectorSize(req.Args) > 10000 {
			return "parallel"
		} else if cf.getVectorSize(req.Args) > 1000 {
			return "simd"
		}
		return "sequential"
	case "matmul":
		return "blas"
	default:
		return "sequential"
	}
}

// getVectorSize estimates the size of vector arguments
func (cf *ComputeFabric) getVectorSize(args []interface{}) int {
	if len(args) == 0 {
		return 0
	}

	if v, ok := args[0].([]float64); ok {
		return len(v)
	}

	if m, ok := args[0].([][]float64); ok {
		return len(m) * len(m[0])
	}

	return 0
}

// GetStats returns fabric statistics
func (cf *ComputeFabric) GetStats() map[string]interface{} {
	cf.mu.RLock()
	defer cf.mu.RUnlock()

	return map[string]interface{}{
		"pending_requests":  len(cf.pending),
		"completed_results": len(cf.completed),
		"active_workers":    cf.workers,
		"agent_loads":       cf.agentLoads,
		"algorithm_stats":   cf.algorithmStats,
	}
}

// GetAgentLoad returns the current load for an agent
func (cf *ComputeFabric) GetAgentLoad(agentID string) int {
	cf.mu.RLock()
	defer cf.mu.RUnlock()

	return cf.agentLoads[agentID]
}

// BalanceLoad redistributes work to balance agent loads
func (cf *ComputeFabric) BalanceLoad() {
	cf.mu.Lock()
	defer cf.mu.Unlock()

	// Find agent with lowest load
	minLoad := int(^uint(0) >> 1) // Max int
	bestAgent := ""

	for agentID, load := range cf.agentLoads {
		if load < minLoad {
			minLoad = load
			bestAgent = agentID
		}
	}

	// Redistribute some pending requests to the best agent
	// This is a simplified implementation
	for _, req := range cf.pending {
		if req.AgentID != bestAgent {
			req.AgentID = bestAgent
			cf.agentLoads[bestAgent]++
		}
	}
}

// Shutdown gracefully shuts down the compute fabric
func (cf *ComputeFabric) Shutdown() {
	close(cf.shutdown)
	cf.wg.Wait()
}

// Helper functions for request generation

// GenerateRequestID creates a unique request ID
func GenerateRequestID(operation string, args []interface{}) string {
	h := sha256.New()
	h.Write([]byte(operation))

	for _, arg := range args {
		switch v := arg.(type) {
		case []float64:
			for _, val := range v {
				binary.Write(h, binary.LittleEndian, val)
			}
		case [][]float64:
			for _, row := range v {
				for _, val := range row {
					binary.Write(h, binary.LittleEndian, val)
				}
			}
		}
	}

	return fmt.Sprintf("%x", h.Sum(nil))
}

// NewCosineRequest creates a cosine similarity request
func NewCosineRequest(v1, v2 []float64, agentID string) *ComputeRequest {
	return &ComputeRequest{
		ID:        GenerateRequestID("cosine", []interface{}{v1, v2}),
		Operation: "cosine",
		Args:      []interface{}{v1, v2},
		Result:    make(chan interface{}, 1),
		Error:     make(chan error, 1),
		Priority:  1,
		AgentID:   agentID,
	}
}

// NewDotRequest creates a dot product request
func NewDotRequest(v1, v2 []float64, agentID string) *ComputeRequest {
	return &ComputeRequest{
		ID:        GenerateRequestID("dot", []interface{}{v1, v2}),
		Operation: "dot",
		Args:      []interface{}{v1, v2},
		Result:    make(chan interface{}, 1),
		Error:     make(chan error, 1),
		Priority:  1,
		AgentID:   agentID,
	}
}

// NewMatMulRequest creates a matrix multiplication request
func NewMatMulRequest(a, b [][]float64, agentID string) *ComputeRequest {
	return &ComputeRequest{
		ID:        GenerateRequestID("matmul", []interface{}{a, b}),
		Operation: "matmul",
		Args:      []interface{}{a, b},
		Result:    make(chan interface{}, 1),
		Error:     make(chan error, 1),
		Priority:  2,
		AgentID:   agentID,
	}
}

// Global compute fabric instance
var globalFabric *ComputeFabric
var fabricOnce sync.Once

// GetGlobalFabric returns the global compute fabric
func GetGlobalFabric() *ComputeFabric {
	fabricOnce.Do(func() {
		globalFabric = NewComputeFabric(10) // Default 10 workers
	})
	return globalFabric
}

// SetGlobalFabricWorkers sets the number of workers in the global fabric
func SetGlobalFabricWorkers(workers int) {
	fabricOnce.Do(func() {
		globalFabric = NewComputeFabric(workers)
	})
}
