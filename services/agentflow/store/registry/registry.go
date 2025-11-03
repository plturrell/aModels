package registry

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// MappingPath resolves the default registry path relative to the repository root.
const MappingPath = "store/langflow_registry.json"

// Registry tracks the Langflow IDs associated with local catalog flows.
type Registry struct {
	path string

	mu    sync.RWMutex
	data  map[string]string
	dirty bool
}

// Load initialises a registry from the supplied path. If the file does not exist,
// a new empty registry is returned.
func Load(path string) (*Registry, error) {
	clean := filepath.Clean(path)
	reg := &Registry{
		path: clean,
		data: map[string]string{},
	}
	fileInfo, err := os.Stat(clean)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return reg, nil
		}
		return nil, fmt.Errorf("stat registry %s: %w", clean, err)
	}
	if fileInfo.IsDir() {
		return nil, fmt.Errorf("registry path %s is a directory", clean)
	}
	content, err := os.ReadFile(clean)
	if err != nil {
		return nil, fmt.Errorf("read registry %s: %w", clean, err)
	}
	if len(content) == 0 {
		return reg, nil
	}
	if err := json.Unmarshal(content, &reg.data); err != nil {
		return nil, fmt.Errorf("decode registry %s: %w", clean, err)
	}
	return reg, nil
}

// Resolve returns the remote Langflow ID for the given local identifier.
func (r *Registry) Resolve(localID string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.data[localID]
}

// Set assigns the remote ID for the given local flow.
func (r *Registry) Set(localID, remoteID string) {
	if localID == "" || remoteID == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.data[localID]; ok && existing == remoteID {
		return
	}
	r.data[localID] = remoteID
	r.dirty = true
}

// Delete removes a mapping. This is best-effort: errors during save should be handled separately.
func (r *Registry) Delete(localID string) {
	if localID == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.data[localID]; ok {
		delete(r.data, localID)
		r.dirty = true
	}
}

// Save persists the registry to disk when it has been modified.
func (r *Registry) Save() error {
	r.mu.RLock()
	dirty := r.dirty
	r.mu.RUnlock()
	if !dirty {
		return nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(r.path), 0o755); err != nil {
		return fmt.Errorf("create registry directory: %w", err)
	}

	payload, err := json.MarshalIndent(r.data, "", "  ")
	if err != nil {
		return fmt.Errorf("encode registry: %w", err)
	}
	tmp := r.path + ".tmp"
	if err := os.WriteFile(tmp, payload, 0o644); err != nil {
		return fmt.Errorf("write registry tmp: %w", err)
	}
	if err := os.Rename(tmp, r.path); err != nil {
		return fmt.Errorf("replace registry: %w", err)
	}
	r.dirty = false
	return nil
}
