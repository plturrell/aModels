//go:build !gguf

package gguf

// Model is a minimal stub to satisfy references when GGUF is disabled.
type Model struct{}

// Load creates a stub model when GGUF is disabled.
func Load(path string, opts ...interface{}) (*Model, error) { // signature loosely matches real
    _ = path
    _ = opts
    return &Model{}, nil
}

// Generate returns empty content to indicate GGUF is not enabled in this build.
func (m *Model) Generate(prompt string, maxTokens int, temperature, topP float64, topK int) (string, int, error) {
    return "", 0, nil
}

// Close is a no-op in the stub.
func (m *Model) Close() {}


