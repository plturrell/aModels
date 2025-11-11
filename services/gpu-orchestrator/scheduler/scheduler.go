package scheduler

import (
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_allocator"
	"github.com/plturrell/aModels/services/gpu-orchestrator/workload_analyzer"
)

// ScheduledRequest represents a request scheduled for GPU allocation
type ScheduledRequest struct {
	RequestID      string
	AllocationReq  *gpu_allocator.AllocationRequest
	WorkloadReq    *workload_analyzer.WorkloadRequirements
	SubmittedAt    time.Time
	Priority       int
}

// Scheduler schedules GPU allocation requests based on priority and availability
type Scheduler struct {
	allocator      *gpu_allocator.GPUAllocator
	workloadAnalyzer *workload_analyzer.WorkloadAnalyzer
	pendingQueue   []*ScheduledRequest
	logger         *log.Logger
}

// NewScheduler creates a new scheduler
func NewScheduler(allocator *gpu_allocator.GPUAllocator, workloadAnalyzer *workload_analyzer.WorkloadAnalyzer, logger *log.Logger) *Scheduler {
	scheduler := &Scheduler{
		allocator:       allocator,
		workloadAnalyzer: workloadAnalyzer,
		pendingQueue:    make([]*ScheduledRequest, 0),
		logger:          logger,
	}

	// Start background scheduling loop
	go scheduler.schedulingLoop()

	return scheduler
}

// Schedule schedules a GPU allocation request
func (s *Scheduler) Schedule(serviceName string, workloadType string, workloadData map[string]interface{}) (*gpu_allocator.Allocation, error) {
	// Analyze workload to get requirements
	workloadReq, err := s.workloadAnalyzer.AnalyzeWorkload(workloadType, workloadData)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze workload: %w", err)
	}

	// Create allocation request
	allocationReq := &gpu_allocator.AllocationRequest{
		ServiceName:    serviceName,
		RequiredGPUs:   workloadReq.RequiredGPUs,
		MinMemoryMB:    workloadReq.MinMemoryMB,
		MaxMemoryMB:    workloadReq.MaxMemoryMB,
		Priority:       workloadReq.Priority,
		MaxUtilization: workloadReq.MaxUtilization,
	}

	if workloadReq.EstimatedDuration != nil {
		ttl := *workloadReq.EstimatedDuration * 2 // Add buffer
		allocationReq.TTL = &ttl
	}

	// Try immediate allocation
	allocation, err := s.allocator.Allocate(allocationReq)
	if err == nil {
		s.logger.Printf("Immediately allocated GPUs for service %s", serviceName)
		return allocation, nil
	}

	// If immediate allocation fails, add to queue
	scheduledReq := &ScheduledRequest{
		RequestID:     fmt.Sprintf("%s-%d", serviceName, time.Now().UnixNano()),
		AllocationReq: allocationReq,
		WorkloadReq:   workloadReq,
		SubmittedAt:   time.Now(),
		Priority:      workloadReq.Priority,
	}

	s.pendingQueue = append(s.pendingQueue, scheduledReq)
	s.logger.Printf("Queued GPU allocation request for service %s (priority: %d)", serviceName, workloadReq.Priority)

	// Sort queue by priority (higher priority first)
	sort.Slice(s.pendingQueue, func(i, j int) bool {
		if s.pendingQueue[i].Priority != s.pendingQueue[j].Priority {
			return s.pendingQueue[i].Priority > s.pendingQueue[j].Priority
		}
		return s.pendingQueue[i].SubmittedAt.Before(s.pendingQueue[j].SubmittedAt)
	})

	// Return pending status
	return nil, fmt.Errorf("queued for allocation: %s", scheduledReq.RequestID)
}

// schedulingLoop processes pending requests
func (s *Scheduler) schedulingLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// Cleanup expired allocations
		s.allocator.CleanupExpired()

		// Try to process pending requests
		var processed []int
		for i, req := range s.pendingQueue {
			allocation, err := s.allocator.Allocate(req.AllocationReq)
			if err == nil {
				s.logger.Printf("Allocated GPUs from queue for service %s (request ID: %s)", req.AllocationReq.ServiceName, req.RequestID)
				processed = append(processed, i)
			}
		}

		// Remove processed requests (in reverse order to maintain indices)
		for i := len(processed) - 1; i >= 0; i-- {
			idx := processed[i]
			s.pendingQueue = append(s.pendingQueue[:idx], s.pendingQueue[idx+1:]...)
		}
	}
}

// GetQueueStatus returns the current queue status
func (s *Scheduler) GetQueueStatus() []*ScheduledRequest {
	return s.pendingQueue
}

