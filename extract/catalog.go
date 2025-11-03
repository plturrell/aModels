package main

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
)

// Project represents a project in the catalog.
type Project struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// System represents a system in the catalog.
type System struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// InformationSystem represents an information system in the catalog.
type InformationSystem struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// Catalog represents the entire catalog of projects, systems, and information systems.
type Catalog struct {
	Projects           []Project           `json:"projects"`
	Systems            []System            `json:"systems"`
	InformationSystems []InformationSystem `json:"information_systems"`

	mu       sync.RWMutex
	filePath string
}

// NewCatalog creates a new catalog and loads it from the specified file.
func NewCatalog(filePath string) (*Catalog, error) {
	c := &Catalog{filePath: filePath}
	if err := c.Load(); err != nil {
		return nil, err
	}
	return c, nil
}

// Load loads the catalog from the file.
func (c *Catalog) Load() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data, err := os.ReadFile(c.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // File doesn't exist yet, that's fine
		}
		return err
	}

	return json.Unmarshal(data, c)
}

// Save saves the catalog to the file.
func (c *Catalog) Save() error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(c.filePath, data, 0644)
}

func (c *Catalog) EnsureProject(id, name string) bool {
	id = strings.TrimSpace(id)
	if id == "" {
		return false
	}
	if name = strings.TrimSpace(name); name == "" {
		name = id
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, project := range c.Projects {
		if project.ID == id {
			return false
		}
	}
	c.Projects = append(c.Projects, Project{ID: id, Name: name})
	return true
}

func (c *Catalog) EnsureSystem(id, name string) bool {
	id = strings.TrimSpace(id)
	if id == "" {
		return false
	}
	if name = strings.TrimSpace(name); name == "" {
		name = id
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, system := range c.Systems {
		if system.ID == id {
			return false
		}
	}
	c.Systems = append(c.Systems, System{ID: id, Name: name})
	return true
}

func (c *Catalog) EnsureInformationSystem(id, name string) bool {
	id = strings.TrimSpace(id)
	if id == "" {
		return false
	}
	if name = strings.TrimSpace(name); name == "" {
		name = id
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, info := range c.InformationSystems {
		if info.ID == id {
			return false
		}
	}
	c.InformationSystems = append(c.InformationSystems, InformationSystem{ID: id, Name: name})
	return true
}
