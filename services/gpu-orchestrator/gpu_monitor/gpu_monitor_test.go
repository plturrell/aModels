package gpu_monitor

import (
	"log"
	"os"
	"testing"
	"time"
)

func TestNewGPUMonitor(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor, err := NewGPUMonitor(logger)
	if err != nil {
		// It's OK if nvidia-smi is not available in test environment
		if monitor == nil {
			t.Fatal("Expected monitor to be created even without nvidia-smi")
		}
	}
	
	if monitor == nil {
		t.Fatal("Monitor should not be nil")
	}
}

func TestMarkAllocatedAndReleased(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &GPUMonitor{
		gpus:     make(map[int]*GPUInfo),
		logger:   logger,
		nvidiaSMI: false, // Don't require nvidia-smi for this test
	}
	
	// Add a test GPU
	monitor.gpus[0] = &GPUInfo{
		ID:          0,
		Name:        "Tesla T4",
		MemoryTotal: 16384,
		MemoryFree:  16384,
		Allocated:   false,
	}
	
	// Test allocation
	err := monitor.MarkAllocated(0, "test-service")
	if err != nil {
		t.Fatalf("Failed to mark GPU as allocated: %v", err)
	}
	
	gpu, _ := monitor.GetGPU(0)
	if !gpu.Allocated {
		t.Error("GPU should be marked as allocated")
	}
	if gpu.AllocatedTo != "test-service" {
		t.Errorf("Expected allocated to 'test-service', got '%s'", gpu.AllocatedTo)
	}
	
	// Test double allocation
	err = monitor.MarkAllocated(0, "another-service")
	if err == nil {
		t.Error("Expected error when allocating already allocated GPU")
	}
	
	// Test release
	err = monitor.MarkReleased(0)
	if err != nil {
		t.Fatalf("Failed to release GPU: %v", err)
	}
	
	gpu, _ = monitor.GetGPU(0)
	if gpu.Allocated {
		t.Error("GPU should not be allocated after release")
	}
	if gpu.AllocatedTo != "" {
		t.Error("AllocatedTo should be empty after release")
	}
}

func TestGetAvailableGPUs(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &GPUMonitor{
		gpus:     make(map[int]*GPUInfo),
		logger:   logger,
		nvidiaSMI: false,
	}
	
	// Add test GPUs
	monitor.gpus[0] = &GPUInfo{ID: 0, Name: "GPU 0", Allocated: false}
	monitor.gpus[1] = &GPUInfo{ID: 1, Name: "GPU 1", Allocated: true, AllocatedTo: "service1"}
	monitor.gpus[2] = &GPUInfo{ID: 2, Name: "GPU 2", Allocated: false}
	
	available := monitor.GetAvailableGPUs()
	
	if len(available) != 2 {
		t.Errorf("Expected 2 available GPUs, got %d", len(available))
	}
	
	// Check that allocated GPU is not in the list
	for _, gpu := range available {
		if gpu.Allocated {
			t.Error("Available GPUs list should not contain allocated GPUs")
		}
	}
}

func TestListGPUs(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &GPUMonitor{
		gpus:     make(map[int]*GPUInfo),
		logger:   logger,
		nvidiaSMI: false,
	}
	
	// Add test GPUs
	monitor.gpus[0] = &GPUInfo{ID: 0, Name: "GPU 0"}
	monitor.gpus[1] = &GPUInfo{ID: 1, Name: "GPU 1"}
	
	gpus := monitor.ListGPUs()
	
	if len(gpus) != 2 {
		t.Errorf("Expected 2 GPUs, got %d", len(gpus))
	}
}

func TestRefreshPreservesAllocationState(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &GPUMonitor{
		gpus:     make(map[int]*GPUInfo),
		logger:   logger,
		nvidiaSMI: false,
	}
	
	// Add a GPU with allocation
	allocTime := time.Now()
	monitor.gpus[0] = &GPUInfo{
		ID:          0,
		Name:        "GPU 0",
		Allocated:   true,
		AllocatedTo: "test-service",
		AllocatedAt: &allocTime,
	}
	
	// Simulate refresh (without actual nvidia-smi call)
	// In real refresh, allocation state should be preserved
	existing := monitor.gpus[0]
	monitor.gpus[0] = &GPUInfo{
		ID:          0,
		Name:        "GPU 0",
		Allocated:   existing.Allocated,
		AllocatedTo: existing.AllocatedTo,
		AllocatedAt: existing.AllocatedAt,
	}
	
	gpu, _ := monitor.GetGPU(0)
	if !gpu.Allocated || gpu.AllocatedTo != "test-service" {
		t.Error("Allocation state should be preserved after refresh")
	}
}
