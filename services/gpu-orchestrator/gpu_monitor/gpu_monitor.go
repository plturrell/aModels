package gpu_monitor

import (
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// GPUInfo represents information about a GPU
type GPUInfo struct {
	ID            int     `json:"id"`
	Name          string  `json:"name"`
	MemoryTotal   int64   `json:"memory_total"`   // in MB
	MemoryUsed    int64   `json:"memory_used"`    // in MB
	MemoryFree    int64   `json:"memory_free"`    // in MB
	Utilization   float64 `json:"utilization"`    // percentage
	Temperature   float64 `json:"temperature"`    // Celsius
	PowerDraw     float64 `json:"power_draw"`     // Watts
	LastUpdated   time.Time `json:"last_updated"`
	Allocated     bool    `json:"allocated"`
	AllocatedTo   string  `json:"allocated_to"`   // service name
	AllocatedAt   *time.Time `json:"allocated_at,omitempty"`
}

// GPUMonitor monitors GPU resources
type GPUMonitor struct {
	gpus     map[int]*GPUInfo
	mu       sync.RWMutex
	logger   *log.Logger
	nvidiaSMI bool
}

// NewGPUMonitor creates a new GPU monitor
func NewGPUMonitor(logger *log.Logger) (*GPUMonitor, error) {
	monitor := &GPUMonitor{
		gpus:     make(map[int]*GPUInfo),
		logger:   logger,
		nvidiaSMI: false,
	}

	// Check if nvidia-smi is available
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		monitor.nvidiaSMI = true
		logger.Println("nvidia-smi found, GPU monitoring enabled")
	} else {
		logger.Println("nvidia-smi not found, GPU monitoring disabled")
		return monitor, nil // Return monitor but without GPU support
	}

	// Initial refresh
	if err := monitor.Refresh(); err != nil {
		return nil, fmt.Errorf("failed to refresh GPU info: %w", err)
	}

	return monitor, nil
}

// Refresh updates GPU information from nvidia-smi
func (m *GPUMonitor) Refresh() error {
	if !m.nvidiaSMI {
		return nil // No-op if nvidia-smi not available
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Query nvidia-smi for GPU information
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits")
	
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("nvidia-smi query failed: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}

		fields := strings.Split(line, ", ")
		if len(fields) < 8 {
			continue
		}

		gpuID, _ := strconv.Atoi(strings.TrimSpace(fields[0]))
		gpuName := strings.TrimSpace(fields[1])
		memTotal, _ := strconv.ParseInt(strings.TrimSpace(fields[2]), 10, 64)
		memUsed, _ := strconv.ParseInt(strings.TrimSpace(fields[3]), 10, 64)
		memFree, _ := strconv.ParseInt(strings.TrimSpace(fields[4]), 10, 64)
		utilization, _ := strconv.ParseFloat(strings.TrimSpace(fields[5]), 64)
		temperature, _ := strconv.ParseFloat(strings.TrimSpace(fields[6]), 64)
		powerDraw, _ := strconv.ParseFloat(strings.TrimSpace(fields[7]), 64)

		// Preserve allocation state if GPU already exists
		existing := m.gpus[gpuID]
		allocated := false
		allocatedTo := ""
		var allocatedAt *time.Time
		
		if existing != nil {
			allocated = existing.Allocated
			allocatedTo = existing.AllocatedTo
			allocatedAt = existing.AllocatedAt
		}

		m.gpus[gpuID] = &GPUInfo{
			ID:          gpuID,
			Name:        gpuName,
			MemoryTotal:  memTotal,
			MemoryUsed:   memUsed,
			MemoryFree:   memFree,
			Utilization:  utilization,
			Temperature:  temperature,
			PowerDraw:    powerDraw,
			LastUpdated:  time.Now(),
			Allocated:    allocated,
			AllocatedTo:  allocatedTo,
			AllocatedAt:  allocatedAt,
		}
	}

	return nil
}

// GetGPU returns information about a specific GPU
func (m *GPUMonitor) GetGPU(id int) (*GPUInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	gpu, ok := m.gpus[id]
	if !ok {
		return nil, fmt.Errorf("GPU %d not found", id)
	}

	return gpu, nil
}

// ListGPUs returns all available GPUs
func (m *GPUMonitor) ListGPUs() []*GPUInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	gpus := make([]*GPUInfo, 0, len(m.gpus))
	for _, gpu := range m.gpus {
		gpus = append(gpus, gpu)
	}

	return gpus
}

// GetAvailableGPUs returns GPUs that are not allocated
func (m *GPUMonitor) GetAvailableGPUs() []*GPUInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	available := make([]*GPUInfo, 0)
	for _, gpu := range m.gpus {
		if !gpu.Allocated {
			available = append(available, gpu)
		}
	}

	return available
}

// MarkAllocated marks a GPU as allocated to a service
func (m *GPUMonitor) MarkAllocated(gpuID int, serviceName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	gpu, ok := m.gpus[gpuID]
	if !ok {
		return fmt.Errorf("GPU %d not found", gpuID)
	}

	if gpu.Allocated {
		return fmt.Errorf("GPU %d is already allocated to %s", gpuID, gpu.AllocatedTo)
	}

	now := time.Now()
	gpu.Allocated = true
	gpu.AllocatedTo = serviceName
	gpu.AllocatedAt = &now

	return nil
}

// MarkReleased marks a GPU as released
func (m *GPUMonitor) MarkReleased(gpuID int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	gpu, ok := m.gpus[gpuID]
	if !ok {
		return fmt.Errorf("GPU %d not found", gpuID)
	}

	gpu.Allocated = false
	gpu.AllocatedTo = ""
	gpu.AllocatedAt = nil

	return nil
}

// IsAvailable returns true if nvidia-smi is available
func (m *GPUMonitor) IsAvailable() bool {
	return m.nvidiaSMI
}

