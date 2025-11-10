package gpu_allocator

import (
	"log"
	"os"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_monitor"
)

func createTestMonitor() *gpu_monitor.GPUMonitor {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	monitor := &gpu_monitor.GPUMonitor{}
	
	// Use reflection or create a test helper in gpu_monitor package
	// For now, we'll create a mock monitor
	return monitor
}

func TestAllocateGPUs(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	// Create mock monitor with test GPUs
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Utilization: 10.0, Allocated: false},
			{ID: 1, Name: "GPU 1", MemoryFree: 8192, Utilization: 20.0, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	req := &AllocationRequest{
		ServiceName:  "test-service",
		RequiredGPUs: 1,
		MinMemoryMB:  4096,
		Priority:     5,
	}
	
	allocation, err := allocator.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate GPUs: %v", err)
	}
	
	if allocation == nil {
		t.Fatal("Allocation should not be nil")
	}
	
	if allocation.ServiceName != "test-service" {
		t.Errorf("Expected service name 'test-service', got '%s'", allocation.ServiceName)
	}
	
	if len(allocation.GPUIDs) != 1 {
		t.Errorf("Expected 1 GPU, got %d", len(allocation.GPUIDs))
	}
}

func TestAllocateInsufficientGPUs(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	req := &AllocationRequest{
		ServiceName:  "test-service",
		RequiredGPUs: 2,
		MinMemoryMB:  4096,
		Priority:     5,
	}
	
	_, err := allocator.Allocate(req)
	if err == nil {
		t.Error("Expected error when requesting more GPUs than available")
	}
}

func TestAllocateMemoryRequirement(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 2048, Allocated: false},
			{ID: 1, Name: "GPU 1", MemoryFree: 8192, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	req := &AllocationRequest{
		ServiceName:  "test-service",
		RequiredGPUs: 1,
		MinMemoryMB:  4096,
		Priority:     5,
	}
	
	allocation, err := allocator.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate GPUs: %v", err)
	}
	
	// Should allocate GPU 1 (has more memory)
	if allocation.GPUIDs[0] != 1 {
		t.Errorf("Expected GPU 1 to be allocated, got GPU %d", allocation.GPUIDs[0])
	}
}

func TestReleaseGPUs(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	req := &AllocationRequest{
		ServiceName:  "test-service",
		RequiredGPUs: 1,
		Priority:     5,
	}
	
	allocation, err := allocator.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate GPUs: %v", err)
	}
	
	// Release the allocation
	err = allocator.Release(allocation.ID)
	if err != nil {
		t.Fatalf("Failed to release GPUs: %v", err)
	}
	
	// Should not be in allocations map anymore
	_, err = allocator.GetAllocation(allocation.ID)
	if err == nil {
		t.Error("Expected error when getting released allocation")
	}
}

func TestReleaseByService(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
			{ID: 1, Name: "GPU 1", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	// Allocate for same service twice
	req1 := &AllocationRequest{ServiceName: "test-service", RequiredGPUs: 1, Priority: 5}
	req2 := &AllocationRequest{ServiceName: "test-service", RequiredGPUs: 1, Priority: 5}
	
	allocator.Allocate(req1)
	allocator.Allocate(req2)
	
	// Release all allocations for the service
	err := allocator.ReleaseByService("test-service")
	if err != nil {
		t.Fatalf("Failed to release by service: %v", err)
	}
	
	allocations := allocator.ListAllocations()
	if len(allocations) != 0 {
		t.Errorf("Expected 0 allocations after release by service, got %d", len(allocations))
	}
}

func TestCleanupExpired(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	// Create allocation with short TTL
	ttl := 1 * time.Millisecond
	req := &AllocationRequest{
		ServiceName:  "test-service",
		RequiredGPUs: 1,
		Priority:     5,
		TTL:          &ttl,
	}
	
	allocation, _ := allocator.Allocate(req)
	
	// Wait for expiration
	time.Sleep(10 * time.Millisecond)
	
	count := allocator.CleanupExpired()
	if count != 1 {
		t.Errorf("Expected 1 expired allocation, got %d", count)
	}
	
	// Allocation should be removed
	_, err := allocator.GetAllocation(allocation.ID)
	if err == nil {
		t.Error("Expected error when getting expired allocation")
	}
}

func TestPreferredGPUs(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	monitor := &mockGPUMonitor{
		gpus: []*gpu_monitor.GPUInfo{
			{ID: 0, Name: "GPU 0", MemoryFree: 16384, Allocated: false},
			{ID: 1, Name: "GPU 1", MemoryFree: 16384, Allocated: false},
			{ID: 2, Name: "GPU 2", MemoryFree: 16384, Allocated: false},
		},
	}
	
	allocator := NewGPUAllocator(monitor, logger)
	
	req := &AllocationRequest{
		ServiceName:   "test-service",
		RequiredGPUs:  1,
		Priority:      5,
		PreferredGPUs: []int{2},
	}
	
	allocation, err := allocator.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate GPUs: %v", err)
	}
	
	// Should prefer GPU 2
	if allocation.GPUIDs[0] != 2 {
		t.Errorf("Expected preferred GPU 2, got GPU %d", allocation.GPUIDs[0])
	}
}

// Mock GPU Monitor for testing
type mockGPUMonitor struct {
	gpus      []*gpu_monitor.GPUInfo
	allocated map[int]string
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
			if gpu.Allocated {
				return nil // Already allocated
			}
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
