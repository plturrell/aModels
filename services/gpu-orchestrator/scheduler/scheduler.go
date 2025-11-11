package scheduler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_allocator"
	"github.com/plturrell/aModels/services/gpu-orchestrator/workload_analyzer"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	queueDepth = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gpu_scheduler_queue_depth",
			Help: "Number of requests waiting in queue",
		},
	)

	queueWaitTime = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "gpu_scheduler_queue_wait_seconds",
			Help:    "Time requests spend waiting in queue",
			Buckets: []float64{1, 5, 10, 30, 60, 120, 300, 600},
		},
	)

	scheduledTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_scheduler_scheduled_total",
			Help: "Total number of scheduled requests by status",
		},
		[]string{"status"},
	)
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
	allocator        *gpu_allocator.GPUAllocator
	workloadAnalyzer *workload_analyzer.WorkloadAnalyzer
	pendingQueue     []*ScheduledRequest
	httpClient       *http.Client
	logger           *log.Logger
}

// NewScheduler creates a new scheduler
func NewScheduler(allocator *gpu_allocator.GPUAllocator, workloadAnalyzer *workload_analyzer.WorkloadAnalyzer, logger *log.Logger) *Scheduler {
	scheduler := &Scheduler{
		allocator:        allocator,
		workloadAnalyzer: workloadAnalyzer,
		pendingQueue:     make([]*ScheduledRequest, 0),
		httpClient:       &http.Client{Timeout: 10 * time.Second},
		logger:           logger,
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

	// Extract webhook URL from workload data
	webhookURL := ""
	if workloadData != nil {
		if url, ok := workloadData["webhook_url"].(string); ok {
			webhookURL = url
		}
	}

	// Create allocation request
	allocationReq := &gpu_allocator.AllocationRequest{
		ServiceName:    serviceName,
		RequiredGPUs:   workloadReq.RequiredGPUs,
		MinMemoryMB:    workloadReq.MinMemoryMB,
		MaxMemoryMB:    workloadReq.MaxMemoryMB,
		Priority:       workloadReq.Priority,
		MaxUtilization: workloadReq.MaxUtilization,
		WebhookURL:     webhookURL,
		WorkloadType:   workloadType,
		WorkloadData:   workloadData,
	}

	if workloadReq.EstimatedDuration != nil {
		ttl := *workloadReq.EstimatedDuration * 2 // Add buffer
		allocationReq.TTL = &ttl
	}

	// Try immediate allocation
	allocation, err := s.allocator.Allocate(allocationReq)
	if err == nil {
		scheduledTotal.WithLabelValues("immediate").Inc()
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
	scheduledTotal.WithLabelValues("queued").Inc()
	queueDepth.Set(float64(len(s.pendingQueue)))
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
				waitTime := time.Since(req.SubmittedAt).Seconds()
				queueWaitTime.Observe(waitTime)
				scheduledTotal.WithLabelValues("allocated_from_queue").Inc()
				s.logger.Printf("Allocated GPUs from queue for service %s (request ID: %s, wait time: %.1fs)", req.AllocationReq.ServiceName, req.RequestID, waitTime)
				processed = append(processed, i)
				
				// Send webhook notification if webhook URL is provided
				if req.AllocationReq.WebhookURL != "" {
					go s.sendWebhookNotification(req.AllocationReq.WebhookURL, allocation, req)
				}
			}
		}

		// Remove processed requests (in reverse order to maintain indices)
		for i := len(processed) - 1; i >= 0; i-- {
			idx := processed[i]
			s.pendingQueue = append(s.pendingQueue[:idx], s.pendingQueue[idx+1:]...)
		}
		
		// Update queue depth metric
		queueDepth.Set(float64(len(s.pendingQueue)))
	}
}

// GetQueueStatus returns the current queue status
func (s *Scheduler) GetQueueStatus() []*ScheduledRequest {
	return s.pendingQueue
}

// sendWebhookNotification sends a webhook notification when an allocation succeeds
func (s *Scheduler) sendWebhookNotification(webhookURL string, allocation *gpu_allocator.Allocation, req *ScheduledRequest) {
	notification := map[string]interface{}{
		"status":       "allocated",
		"request_id":   req.RequestID,
		"allocation":   allocation,
		"submitted_at": req.SubmittedAt,
		"allocated_at": allocation.AllocatedAt,
		"wait_time_seconds": allocation.AllocatedAt.Sub(req.SubmittedAt).Seconds(),
	}
	
	jsonData, err := json.Marshal(notification)
	if err != nil {
		s.logger.Printf("Failed to marshal webhook notification: %v", err)
		return
	}
	
	resp, err := s.httpClient.Post(webhookURL, "application/json", bytes.NewReader(jsonData))
	if err != nil {
		s.logger.Printf("Failed to send webhook notification to %s: %v", webhookURL, err)
		return
	}
	defer resp.Body.Close()
	
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		s.logger.Printf("Webhook notification sent successfully to %s for request %s", webhookURL, req.RequestID)
	} else {
		s.logger.Printf("Webhook notification to %s returned status %d for request %s", webhookURL, resp.StatusCode, req.RequestID)
	}
}

