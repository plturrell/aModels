package scheduler

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_allocator"
	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_monitor"
	"github.com/plturrell/aModels/services/gpu-orchestrator/workload_analyzer"
)

func TestScheduleImmediate(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := gpu_allocator.NewGPUAllocator(monitor, logger)
	analyzer := workload_analyzer.NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	// Don't start background loop for this test
	scheduler := &Scheduler{
		allocator:        allocator,
		workloadAnalyzer: analyzer,
		pendingQueue:     make([]*ScheduledRequest, 0),
		httpClient:       &http.Client{Timeout: 10 * time.Second},
		logger:           logger,
	}
	
	allocation, err := scheduler.Schedule("test-service", "inference", map[string]interface{}{
		"model_size": "small",
	})
	
	if err != nil {
		t.Fatalf("Expected immediate allocation, got error: %v", err)
	}
	
	if allocation == nil {
		t.Fatal("Expected allocation to be returned")
	}
}

func TestScheduleQueued(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	// No GPUs available
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{},
	}
	
	allocator := gpu_allocator.NewGPUAllocator(monitor, logger)
	analyzer := workload_analyzer.NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	scheduler := &Scheduler{
		allocator:        allocator,
		workloadAnalyzer: analyzer,
		pendingQueue:     make([]*ScheduledRequest, 0),
		httpClient:       &http.Client{Timeout: 10 * time.Second},
		logger:           logger,
	}
	
	_, err := scheduler.Schedule("test-service", "inference", map[string]interface{}{})
	
	if err == nil {
		t.Error("Expected error when GPUs not available")
	}
	
	// Should be queued
	queue := scheduler.GetQueueStatus()
	if len(queue) != 1 {
		t.Errorf("Expected 1 queued request, got %d", len(queue))
	}
}

func TestWebhookNotification(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	// Create test HTTP server to receive webhook
	webhookReceived := make(chan bool, 1)
	var receivedPayload map[string]interface{}
	
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedPayload)
		w.WriteHeader(http.StatusOK)
		webhookReceived <- true
	}))
	defer server.Close()
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := gpu_allocator.NewGPUAllocator(monitor, logger)
	analyzer := workload_analyzer.NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	scheduler := NewScheduler(allocator, analyzer, logger)
	
	// Schedule with webhook URL
	scheduler.Schedule("test-service", "inference", map[string]interface{}{
		"webhook_url": server.URL,
		"model_size":  "small",
	})
	
	// Wait for webhook notification (should be immediate since GPU is available)
	// Note: In real scenario, webhook is only sent when allocation comes from queue
	// This test verifies the mechanism works
	
	// For queued allocation test, we need to simulate GPU becoming available
	monitor.gpus = []*gpu_monitor.GPUInfo{}
	_, err := scheduler.Schedule("test-service-2", "inference", map[string]interface{}{
		"webhook_url": server.URL,
		"model_size":  "small",
	})
	
	if err == nil {
		t.Error("Expected error when no GPUs available")
	}
	
	// Now make GPU available
	monitor.gpus = []*gpu_monitor.GPUInfo{
		{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
	}
	
	// Wait for background scheduler to process queue
	select {
	case <-webhookReceived:
		if receivedPayload["status"] != "allocated" {
			t.Error("Expected status 'allocated' in webhook payload")
		}
		if receivedPayload["request_id"] == nil {
			t.Error("Expected request_id in webhook payload")
		}
	case <-time.After(10 * time.Second):
		t.Error("Webhook notification not received within timeout")
	}
}

func TestPriorityOrdering(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{},
	}
	
	allocator := gpu_allocator.NewGPUAllocator(monitor, logger)
	analyzer := workload_analyzer.NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	scheduler := &Scheduler{
		allocator:        allocator,
		workloadAnalyzer: analyzer,
		pendingQueue:     make([]*ScheduledRequest, 0),
		httpClient:       &http.Client{Timeout: 10 * time.Second},
		logger:           logger,
	}
	
	// Queue multiple requests with different priorities
	scheduler.Schedule("low-priority", "generic", map[string]interface{}{"priority": 3})
	scheduler.Schedule("high-priority", "inference", map[string]interface{}{}) // inference has priority 7
	scheduler.Schedule("medium-priority", "training", map[string]interface{}{}) // training has priority 5
	
	queue := scheduler.GetQueueStatus()
	
	if len(queue) != 3 {
		t.Fatalf("Expected 3 queued requests, got %d", len(queue))
	}
	
	// Check priority ordering (highest first)
	if queue[0].Priority < queue[1].Priority {
		t.Error("Queue should be ordered by priority (highest first)")
	}
}

// Mock GPU Monitor for testing
type mockGPUMonitor struct {
	gpus []*gpu_monitor.GPUInfo
}

func (m *mockGPUMonitor) GetAvailableGPUs() []*gpu_monitor.GPUInfo {
	available := make([]*gpu_monitor.GPUInfo, 0)
	for _, gpu := range m.gpus {
		if !gpu.Allocated {
			available = append(available, gpu)
		}
	}
	return available
}

func (m *mockGPUMonitor) MarkAllocated(gpuID int, serviceName string) error {
	for _, gpu := range m.gpus {
		if gpu.ID == gpuID {
			gpu.Allocated = true
			gpu.AllocatedTo = serviceName
			return nil
		}
	}
	return nil
}

func (m *mockGPUMonitor) MarkReleased(gpuID int) error {
	for _, gpu := range m.gpus {
		if gpu.ID == gpuID {
			gpu.Allocated = false
			gpu.AllocatedTo = ""
			return nil
		}
	}
	return nil
}

func (m *mockGPUMonitor) IsAvailable() bool {
	return true
}
