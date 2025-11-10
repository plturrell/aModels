package workload_analyzer

import (
	"log"
	"os"
	"testing"
)

func TestAnalyzeTrainingWorkload(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	tests := []struct {
		name          string
		data          map[string]interface{}
		expectedGPUs  int
		expectedMemMB int64
	}{
		{
			name:          "Small model",
			data:          map[string]interface{}{"model_size": "small"},
			expectedGPUs:  1,
			expectedMemMB: 4096,
		},
		{
			name:          "Medium model",
			data:          map[string]interface{}{"model_size": "medium"},
			expectedGPUs:  2,
			expectedMemMB: 8192,
		},
		{
			name:          "Large model",
			data:          map[string]interface{}{"model_size": "large"},
			expectedGPUs:  4,
			expectedMemMB: 16384,
		},
		{
			name:          "XLarge model",
			data:          map[string]interface{}{"model_size": "xlarge"},
			expectedGPUs:  8,
			expectedMemMB: 32768,
		},
		{
			name:          "Large batch size",
			data:          map[string]interface{}{"batch_size": float64(256)},
			expectedGPUs:  4,
			expectedMemMB: 8192,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := analyzer.analyzeTrainingWorkload(tt.data)
			if err != nil {
				t.Fatalf("Failed to analyze workload: %v", err)
			}
			
			if req.RequiredGPUs != tt.expectedGPUs {
				t.Errorf("Expected %d GPUs, got %d", tt.expectedGPUs, req.RequiredGPUs)
			}
			
			if req.MinMemoryMB != tt.expectedMemMB {
				t.Errorf("Expected %d MB memory, got %d", tt.expectedMemMB, req.MinMemoryMB)
			}
		})
	}
}

func TestAnalyzeInferenceWorkload(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	tests := []struct {
		name          string
		data          map[string]interface{}
		expectedGPUs  int
		expectedMemMB int64
		expectedPri   int
	}{
		{
			name:          "Gemma 7B model",
			data:          map[string]interface{}{"model_name": "gemma-7b"},
			expectedGPUs:  1,
			expectedMemMB: 16384,
			expectedPri:   8,
		},
		{
			name:          "Gemma 2B model",
			data:          map[string]interface{}{"model_name": "gemma-2b"},
			expectedGPUs:  1,
			expectedMemMB: 4096,
			expectedPri:   7,
		},
		{
			name:          "VaultGemma model",
			data:          map[string]interface{}{"model_name": "vaultgemma-1b"},
			expectedGPUs:  1,
			expectedMemMB: 2048,
			expectedPri:   6,
		},
		{
			name:          "Phi-3.5 model",
			data:          map[string]interface{}{"model_name": "phi-3.5-mini"},
			expectedGPUs:  1,
			expectedMemMB: 4096,
			expectedPri:   7,
		},
		{
			name:          "High concurrent requests",
			data:          map[string]interface{}{"concurrent_requests": float64(25)},
			expectedGPUs:  4,
			expectedMemMB: 4096,
			expectedPri:   7,
		},
		{
			name:          "Dedicated allocation",
			data:          map[string]interface{}{"dedicated": true},
			expectedGPUs:  1,
			expectedMemMB: 8192,
			expectedPri:   8,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := analyzer.analyzeInferenceWorkload(tt.data)
			if err != nil {
				t.Fatalf("Failed to analyze workload: %v", err)
			}
			
			if req.RequiredGPUs != tt.expectedGPUs {
				t.Errorf("Expected %d GPUs, got %d", tt.expectedGPUs, req.RequiredGPUs)
			}
			
			if req.MinMemoryMB != tt.expectedMemMB {
				t.Errorf("Expected %d MB memory, got %d", tt.expectedMemMB, req.MinMemoryMB)
			}
			
			if req.Priority != tt.expectedPri {
				t.Errorf("Expected priority %d, got %d", tt.expectedPri, req.Priority)
			}
		})
	}
}

func TestAnalyzeEmbeddingWorkload(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	tests := []struct {
		name         string
		data         map[string]interface{}
		expectedGPUs int
	}{
		{
			name:         "Small batch",
			data:         map[string]interface{}{"batch_size": float64(16)},
			expectedGPUs: 1,
		},
		{
			name:         "Large batch",
			data:         map[string]interface{}{"batch_size": float64(64)},
			expectedGPUs: 2,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := analyzer.analyzeEmbeddingWorkload(tt.data)
			if err != nil {
				t.Fatalf("Failed to analyze workload: %v", err)
			}
			
			if req.RequiredGPUs != tt.expectedGPUs {
				t.Errorf("Expected %d GPUs, got %d", tt.expectedGPUs, req.RequiredGPUs)
			}
		})
	}
}

func TestAnalyzeOCRWorkload(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	tests := []struct {
		name         string
		data         map[string]interface{}
		expectedGPUs int
	}{
		{
			name:         "Small image count",
			data:         map[string]interface{}{"image_count": float64(50)},
			expectedGPUs: 1,
		},
		{
			name:         "Large image count",
			data:         map[string]interface{}{"image_count": float64(200)},
			expectedGPUs: 2,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := analyzer.analyzeOCRWorkload(tt.data)
			if err != nil {
				t.Fatalf("Failed to analyze workload: %v", err)
			}
			
			if req.RequiredGPUs != tt.expectedGPUs {
				t.Errorf("Expected %d GPUs, got %d", tt.expectedGPUs, req.RequiredGPUs)
			}
		})
	}
}

func TestAnalyzeGraphProcessingWorkload(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	tests := []struct {
		name          string
		data          map[string]interface{}
		expectedMemMB int64
	}{
		{
			name:          "Small graph",
			data:          map[string]interface{}{"node_count": float64(100000)},
			expectedMemMB: 4096,
		},
		{
			name:          "Large graph",
			data:          map[string]interface{}{"node_count": float64(2000000)},
			expectedMemMB: 8192,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := analyzer.analyzeGraphProcessingWorkload(tt.data)
			if err != nil {
				t.Fatalf("Failed to analyze workload: %v", err)
			}
			
			if req.MinMemoryMB != tt.expectedMemMB {
				t.Errorf("Expected %d MB memory, got %d", tt.expectedMemMB, req.MinMemoryMB)
			}
		})
	}
}

func TestAnalyzeGenericWorkload(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	data := map[string]interface{}{
		"required_gpus":  float64(2),
		"min_memory_mb":  float64(8192),
		"priority":       float64(7),
	}
	
	req, err := analyzer.analyzeGenericWorkload(data)
	if err != nil {
		t.Fatalf("Failed to analyze workload: %v", err)
	}
	
	if req.RequiredGPUs != 2 {
		t.Errorf("Expected 2 GPUs, got %d", req.RequiredGPUs)
	}
	
	if req.MinMemoryMB != 8192 {
		t.Errorf("Expected 8192 MB memory, got %d", req.MinMemoryMB)
	}
	
	if req.Priority != 7 {
		t.Errorf("Expected priority 7, got %d", req.Priority)
	}
}

func TestAnalyzeWorkloadDispatch(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	workloadTypes := []string{"training", "inference", "embedding", "ocr", "graph_processing", "generic"}
	
	for _, wt := range workloadTypes {
		t.Run(wt, func(t *testing.T) {
			req, err := analyzer.AnalyzeWorkload(wt, map[string]interface{}{})
			if err != nil {
				t.Fatalf("Failed to analyze %s workload: %v", wt, err)
			}
			
			if req == nil {
				t.Error("Expected non-nil requirements")
			}
			
			if req.RequiredGPUs < 1 {
				t.Error("Expected at least 1 GPU")
			}
		})
	}
}

func TestWorkflowPriorityExtraction(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	analyzer := NewWorkloadAnalyzer("http://localhost:8081", logger)
	
	data := map[string]interface{}{
		"workflow_priority": float64(9),
	}
	
	req, err := analyzer.analyzeInferenceWorkload(data)
	if err != nil {
		t.Fatalf("Failed to analyze workload: %v", err)
	}
	
	if req.Priority != 9 {
		t.Errorf("Expected priority 9 from workflow context, got %d", req.Priority)
	}
}
