//go:build gguf

package gguf

import (
	"fmt"
	"runtime"
	"strings"
	"sync"

	llama "github.com/go-skynet/go-llama.cpp"
)

// Model wraps a quantized GGUF model loaded via go-llama.cpp.
type Model struct {
	llama *llama.LLama
	mu    sync.Mutex
}

// Load initializes a GGUF model from disk for CPU-only inference.
func Load(path string, opts ...llama.ModelOption) (*Model, error) {
	modelOpts := []llama.ModelOption{
		llama.SetContext(2048),
		llama.SetNBatch(512),
		llama.SetMMap(true),
	}
	modelOpts = append(modelOpts, opts...)

	ll, err := llama.New(path, modelOpts...)
	if err != nil {
		return nil, fmt.Errorf("gguf load failed: %w", err)
	}

	return &Model{
		llama: ll,
	}, nil
}

// Close releases native resources.
func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.llama != nil {
		m.llama.Free()
		m.llama = nil
	}
}

// Generate produces text using the quantized model.
func (m *Model) Generate(prompt string, maxTokens int, temperature float64, topP float64, topK int) (string, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.llama == nil {
		return "", 0, fmt.Errorf("model not loaded")
	}

	if maxTokens <= 0 {
		maxTokens = 128
	}
	if temperature <= 0 {
		temperature = 0.7
	}
	if topP <= 0 || topP > 1 {
		topP = 0.9
	}
	if topK <= 0 {
		topK = 40
	}

	opts := []llama.PredictOption{
		llama.SetTokens(maxTokens),
		llama.SetTemperature(float32(temperature)),
		llama.SetTopP(float32(topP)),
		llama.SetTopK(topK),
		llama.SetThreads(runtime.NumCPU()),
	}

	result, err := m.llama.Predict(prompt, opts...)
	if err != nil {
		return "", 0, err
	}

	tokens := len(strings.Fields(result))
	return strings.TrimSpace(result), tokens, nil
}
