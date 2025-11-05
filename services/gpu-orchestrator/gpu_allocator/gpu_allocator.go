package gpu_allocator

import (
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_monitor"
)

// AllocationRequest represents a request to allocate GPU resources
type AllocationRequest struct {
	ServiceName    string   `json:"service_name"`
	RequiredGPUs   int      `json:"required_gpus"`
	MinMemoryMB    int64    `json:"min_memory_mb,omitempty"`
	MaxMemoryMB    int64    `json:"max_memory_mb,omitempty"`
	Priority       int      `json:"priority"` // Higher priority = more important
	MaxUtilization float64  `json:"max_utilization,omitempty"` // Maximum allowed utilization
	PreferredGPUs  []int    `json:"preferred_gpus,omitempty"` // Preferred GPU IDs
	TTL            *time.Duration `json:"ttl,omitempty"` // Time to live for allocation
}

// Allocation represents an active GPU allocation
type Allocation struct {
	ID            string    `json:"id"`
	ServiceName   string    `json:"service_name"`
	GPUIDs        []int     `json:"gpu_ids"`
	AllocatedAt   time.Time `json:"allocated_at"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
	Priority      int       `json:"priority"`
}

// GPUAllocator manages GPU allocation and deallocation
type GPUAllocator struct {
	monitor      *gpu_monitor.GPUMonitor
	allocations  map[string]*Allocation
	logger       *log.Logger
}

// NewGPUAllocator creates a new GPU allocator
func NewGPUAllocator(monitor *gpu_monitor.GPUMonitor, logger *log.Logger) *GPUAllocator {
	return &GPUAllocator{
		monitor:     monitor,
		allocations: make(map[string]*Allocation),
		logger:      logger,
	}
}

// Allocate allocates GPUs to a service based on the request
func (a *GPUAllocator) Allocate(req *AllocationRequest) (*Allocation, error) {
	if a.monitor == nil || !a.monitor.IsAvailable() {
		return nil, fmt.Errorf("GPU monitoring not available")
	}

	// Get available GPUs
	availableGPUs := a.monitor.GetAvailableGPUs()
	if len(availableGPUs) < req.RequiredGPUs {
		return nil, fmt.Errorf("insufficient GPUs: requested %d, available %d", req.RequiredGPUs, len(availableGPUs))
	}

	// Filter GPUs based on requirements
	candidates := a.filterGPUs(availableGPUs, req)
	if len(candidates) < req.RequiredGPUs {
		return nil, fmt.Errorf("insufficient GPUs meeting requirements: requested %d, available %d", req.RequiredGPUs, len(candidates))
	}

	// Sort candidates by preference (preferred GPUs first, then by memory, then by utilization)
	sort.Slice(candidates, func(i, j int) bool {
		// Prefer specified GPUs
		iPreferred := contains(req.PreferredGPUs, candidates[i].ID)
		jPreferred := contains(req.PreferredGPUs, candidates[j].ID)
		if iPreferred != jPreferred {
			return iPreferred
		}
		// Prefer GPUs with more free memory
		if candidates[i].MemoryFree != candidates[j].MemoryFree {
			return candidates[i].MemoryFree > candidates[j].MemoryFree
		}
		// Prefer GPUs with lower utilization
		return candidates[i].Utilization < candidates[j].Utilization
	})

	// Select the best GPUs
	selectedGPUs := candidates[:req.RequiredGPUs]
	gpuIDs := make([]int, len(selectedGPUs))
	for i, gpu := range selectedGPUs {
		gpuIDs[i] = gpu.ID
		// Mark GPU as allocated
		if err := a.monitor.MarkAllocated(gpu.ID, req.ServiceName); err != nil {
			// Rollback already allocated GPUs
			for _, allocatedID := range gpuIDs[:i] {
				a.monitor.MarkReleased(allocatedID)
			}
			return nil, fmt.Errorf("failed to allocate GPU %d: %w", gpu.ID, err)
		}
	}

	// Create allocation
	allocationID := fmt.Sprintf("%s-%d", req.ServiceName, time.Now().UnixNano())
	allocation := &Allocation{
		ID:          allocationID,
		ServiceName: req.ServiceName,
		GPUIDs:      gpuIDs,
		AllocatedAt: time.Now(),
		Priority:    req.Priority,
	}

	if req.TTL != nil {
		expiresAt := time.Now().Add(*req.TTL)
		allocation.ExpiresAt = &expiresAt
	}

	a.allocations[allocationID] = allocation

	a.logger.Printf("Allocated GPUs %v to service %s (allocation ID: %s)", gpuIDs, req.ServiceName, allocationID)

	return allocation, nil
}

// Release releases GPUs from an allocation
func (a *GPUAllocator) Release(allocationID string) error {
	allocation, ok := a.allocations[allocationID]
	if !ok {
		return fmt.Errorf("allocation %s not found", allocationID)
	}

	// Release all GPUs
	for _, gpuID := range allocation.GPUIDs {
		if err := a.monitor.MarkReleased(gpuID); err != nil {
			a.logger.Printf("Warning: Failed to release GPU %d: %v", gpuID, err)
		}
	}

	delete(a.allocations, allocationID)
	a.logger.Printf("Released allocation %s (GPUs: %v)", allocationID, allocation.GPUIDs)

	return nil
}

// ReleaseByService releases all allocations for a service
func (a *GPUAllocator) ReleaseByService(serviceName string) error {
	var toRelease []string
	for id, allocation := range a.allocations {
		if allocation.ServiceName == serviceName {
			toRelease = append(toRelease, id)
		}
	}

	var errors []error
	for _, id := range toRelease {
		if err := a.Release(id); err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("failed to release some allocations: %v", errors)
	}

	return nil
}

// GetAllocation returns an allocation by ID
func (a *GPUAllocator) GetAllocation(allocationID string) (*Allocation, error) {
	allocation, ok := a.allocations[allocationID]
	if !ok {
		return nil, fmt.Errorf("allocation %s not found", allocationID)
	}

	return allocation, nil
}

// ListAllocations returns all active allocations
func (a *GPUAllocator) ListAllocations() []*Allocation {
	allocations := make([]*Allocation, 0, len(a.allocations))
	for _, allocation := range a.allocations {
		allocations = append(allocations, allocation)
	}

	return allocations
}

// CleanupExpired removes expired allocations
func (a *GPUAllocator) CleanupExpired() int {
	now := time.Now()
	var expired []string

	for id, allocation := range a.allocations {
		if allocation.ExpiresAt != nil && now.After(*allocation.ExpiresAt) {
			expired = append(expired, id)
		}
	}

	count := 0
	for _, id := range expired {
		if err := a.Release(id); err != nil {
			a.logger.Printf("Warning: Failed to cleanup expired allocation %s: %v", id, err)
		} else {
			count++
		}
	}

	return count
}

// filterGPUs filters GPUs based on allocation requirements
func (a *GPUAllocator) filterGPUs(gpus []*gpu_monitor.GPUInfo, req *AllocationRequest) []*gpu_monitor.GPUInfo {
	filtered := make([]*gpu_monitor.GPUInfo, 0)

	for _, gpu := range gpus {
		// Check memory requirements
		if req.MinMemoryMB > 0 && gpu.MemoryFree < req.MinMemoryMB {
			continue
		}
		if req.MaxMemoryMB > 0 && gpu.MemoryFree > req.MaxMemoryMB {
			continue
		}

		// Check utilization requirements
		if req.MaxUtilization > 0 && gpu.Utilization > req.MaxUtilization {
			continue
		}

		filtered = append(filtered, gpu)
	}

	return filtered
}

// contains checks if a slice contains a value
func contains(slice []int, value int) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}

