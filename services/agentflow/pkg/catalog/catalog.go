package catalog

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Spec represents a local Langflow flow definition stored on disk.
type Spec struct {
	ID          string
	Name        string
	Description string
	Category    string
	Tags        []string
	Path        string
	Raw         json.RawMessage
	metadata    map[string]any
}

// Metadata returns additional metadata bundled within the spec definition.
func (s Spec) Metadata(key string) (any, bool) {
	if s.metadata == nil {
		return nil, false
	}
	value, ok := s.metadata[key]
	return value, ok
}

// InjectMetadata ensures the flow spec includes the provided metadata key/value pair.
func (s *Spec) InjectMetadata(key string, value any) error {
	if s == nil {
		return fmt.Errorf("spec is nil")
	}
	var raw map[string]any
	if err := json.Unmarshal(s.Raw, &raw); err != nil {
		return fmt.Errorf("decode spec: %w", err)
	}
	metadata, _ := raw["metadata"].(map[string]any)
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata[key] = value
	raw["metadata"] = metadata

	encoded, err := json.Marshal(raw)
	if err != nil {
		return fmt.Errorf("encode spec metadata: %w", err)
	}
	s.Raw = json.RawMessage(encoded)
	s.metadata = metadata
	return nil
}

// Loader enumerates and retrieves flow specs from a directory tree.
type Loader struct {
	root string
}

// NewLoader initialises a catalog loader rooted at the provided directory.
func NewLoader(root string) *Loader {
	return &Loader{root: filepath.Clean(root)}
}

// Root returns the configured catalog directory.
func (l *Loader) Root() string {
	return l.root
}

// List returns all flow specs discovered under the loader root.
func (l *Loader) List() ([]Spec, error) {
	root := l.root
	if root == "" {
		return nil, errors.New("catalog root directory is not configured")
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("stat catalog root: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("catalog root %s is not a directory", root)
	}

	specs := []Spec{}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(entry.Name()), ".json") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read flow %s: %w", path, err)
		}

		var raw map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return fmt.Errorf("decode flow %s: %w", path, err)
		}

		spec := Spec{
			Path: path,
			Raw:  json.RawMessage(data),
		}

		if id, ok := raw["id"].(string); ok {
			spec.ID = id
		}
		if name, ok := raw["name"].(string); ok {
			spec.Name = name
		}
		if desc, ok := raw["description"].(string); ok {
			spec.Description = desc
		}
		if category, ok := raw["category"].(string); ok {
			spec.Category = category
		}
		if tags, ok := raw["tags"].([]any); ok {
			spec.Tags = make([]string, 0, len(tags))
			for _, tag := range tags {
				if tagStr, ok := tag.(string); ok {
					spec.Tags = append(spec.Tags, tagStr)
				}
			}
		}
		if metadata, ok := raw["metadata"].(map[string]any); ok {
			spec.metadata = metadata
		}

		if spec.ID == "" {
			return fmt.Errorf("flow %s missing 'id' field", path)
		}

		specs = append(specs, spec)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return specs, nil
}

// Load returns the spec for the supplied flow identifier.
func (l *Loader) Load(flowID string) (Spec, error) {
	specs, err := l.List()
	if err != nil {
		return Spec{}, err
	}
	for _, spec := range specs {
		if spec.ID == flowID {
			return spec, nil
		}
	}
	return Spec{}, fmt.Errorf("flow %s not found in catalog %s", flowID, l.root)
}
