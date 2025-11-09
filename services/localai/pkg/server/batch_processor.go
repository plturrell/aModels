package server

import (
	"context"
	"sync"
	"time"
)

// BatchProcessor handles batching of multiple requests for efficient processing
type BatchProcessor struct {
	batchSize    int
	batchTimeout time.Duration
	batches      chan *RequestBatch
	processor    func(context.Context, []*ChatRequest) []*ChatResponse
	mu           sync.Mutex
}

// RequestBatch represents a batch of requests
type RequestBatch struct {
	Requests  []*BatchedRequest
	Results   chan []*ChatResponse
	CreatedAt time.Time
}

// BatchedRequest wraps a chat request with a result channel
type BatchedRequest struct {
	Request *ChatRequest
	Result  chan *ChatResponse
	Error   chan error
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(batchSize int, batchTimeout time.Duration) *BatchProcessor {
	if batchSize <= 0 {
		batchSize = 10 // Default batch size
	}
	if batchTimeout <= 0 {
		batchTimeout = 50 * time.Millisecond // Default timeout
	}

	bp := &BatchProcessor{
		batchSize:    batchSize,
		batchTimeout: batchTimeout,
		batches:      make(chan *RequestBatch, 100),
	}

	// Start batch processor goroutine
	go bp.processBatches()

	return bp
}

// ProcessRequest adds a request to a batch and returns the result
func (bp *BatchProcessor) ProcessRequest(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	batchedReq := &BatchedRequest{
		Request: req,
		Result:  make(chan *ChatResponse, 1),
		Error:   make(chan error, 1),
	}

	// Find or create a batch
	batch := bp.getOrCreateBatch()

	bp.mu.Lock()
	batch.Requests = append(batch.Requests, batchedReq)
	batchSize := len(batch.Requests)
	shouldProcess := batchSize >= bp.batchSize
	bp.mu.Unlock()

	// If batch is full, trigger processing
	if shouldProcess {
		select {
		case bp.batches <- batch:
		default:
		}
	}

	// Wait for result
	select {
	case resp := <-batchedReq.Result:
		return resp, nil
	case err := <-batchedReq.Error:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// getOrCreateBatch gets the current batch or creates a new one
func (bp *BatchProcessor) getOrCreateBatch() *RequestBatch {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	// For simplicity, always create a new batch
	// In a production system, you'd maintain a current batch
	return &RequestBatch{
		Requests:  make([]*BatchedRequest, 0, bp.batchSize),
		Results:   make(chan []*ChatResponse, 1),
		CreatedAt: time.Now(),
	}
}

// processBatches processes batches as they arrive
func (bp *BatchProcessor) processBatches() {
	for batch := range bp.batches {
		go bp.processBatch(batch)
	}
}

// processBatch processes a single batch
func (bp *BatchProcessor) processBatch(batch *RequestBatch) {
	ctx, cancel := context.WithTimeout(context.Background(), RequestTimeoutDefault)
	defer cancel()

	// Extract requests
	requests := make([]*ChatRequest, len(batch.Requests))
	for i, br := range batch.Requests {
		requests[i] = br.Request
	}

	// Process batch (this would call the actual processing function)
	// For now, process individually
	responses := make([]*ChatResponse, len(requests))
	for i, req := range requests {
		// In a real implementation, this would batch process
		// For now, we'll just create a placeholder response
		responses[i] = &ChatResponse{
			ID:      "batch-resp",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
		}
	}

	// Send results back to individual requests
	for i, br := range batch.Requests {
		if i < len(responses) {
			select {
			case br.Result <- responses[i]:
			default:
			}
		} else {
			select {
			case br.Error <- context.DeadlineExceeded:
			default:
			}
		}
	}
}

// SetProcessor sets the batch processing function
func (bp *BatchProcessor) SetProcessor(processor func(context.Context, []*ChatRequest) []*ChatResponse) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.processor = processor
}

// GetStats returns batch processor statistics
func (bp *BatchProcessor) GetStats() map[string]interface{} {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	return map[string]interface{}{
		"batch_size":     bp.batchSize,
		"batch_timeout": bp.batchTimeout.String(),
		"pending_batches": len(bp.batches),
	}
}

