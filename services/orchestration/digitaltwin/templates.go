package digitaltwin

import (
	"context"
	"fmt"
	"log"
)

// TwinTemplate represents a reusable digital twin template.
type TwinTemplate struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Type        string                 `json:"type"`
	Configuration TwinConfiguration    `json:"configuration"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// TwinTemplateManager manages digital twin templates.
type TwinTemplateManager struct {
	templates map[string]*TwinTemplate
	logger    *log.Logger
}

// NewTwinTemplateManager creates a new twin template manager.
func NewTwinTemplateManager(logger *log.Logger) *TwinTemplateManager {
	return &TwinTemplateManager{
		templates: make(map[string]*TwinTemplate),
		logger:    logger,
	}
}

// CreateTemplate creates a new twin template.
func (ttm *TwinTemplateManager) CreateTemplate(template *TwinTemplate) error {
	if template.ID == "" {
		template.ID = fmt.Sprintf("template-%d", len(ttm.templates)+1)
	}

	ttm.templates[template.ID] = template
	if ttm.logger != nil {
		ttm.logger.Printf("Created twin template: %s (%s)", template.ID, template.Name)
	}

	return nil
}

// GetTemplate retrieves a template by ID.
func (ttm *TwinTemplateManager) GetTemplate(id string) (*TwinTemplate, error) {
	template, exists := ttm.templates[id]
	if !exists {
		return nil, fmt.Errorf("template not found: %s", id)
	}

	return template, nil
}

// ListTemplates lists all templates.
func (ttm *TwinTemplateManager) ListTemplates() []*TwinTemplate {
	templates := make([]*TwinTemplate, 0, len(ttm.templates))
	for _, template := range ttm.templates {
		templates = append(templates, template)
	}
	return templates
}

// CreateTwinFromTemplate creates a digital twin from a template.
func (ttm *TwinTemplateManager) CreateTwinFromTemplate(ctx context.Context, templateID, sourceID, name string, manager *TwinManager) (*Twin, error) {
	template, err := ttm.GetTemplate(templateID)
	if err != nil {
		return nil, fmt.Errorf("failed to get template: %w", err)
	}

	req := CreateTwinRequest{
		Name:     name,
		Type:     template.Type,
		SourceID: sourceID,
		Configuration: template.Configuration,
		Metadata: mergeMetadata(template.Metadata, map[string]interface{}{
			"template_id": templateID,
		}),
	}

	return manager.CreateTwin(ctx, req)
}

// RegisterDefaultTemplates registers default templates.
func (ttm *TwinTemplateManager) RegisterDefaultTemplates() {
	// Data Product Template
	dataProductTemplate := &TwinTemplate{
		ID:          "template-data-product",
		Name:        "Data Product Template",
		Description: "Standard template for data product digital twins",
		Type:        "data_product",
		Configuration: TwinConfiguration{
			ReplicationLevel: 0.8,
			SimulationMode:   "full",
			DataGeneration: DataGenerationConfig{
				Strategy:    "synthetic",
				Volume:      10000,
				Distribution: "normal",
			},
		},
		Metadata: map[string]interface{}{
			"category": "data_product",
			"default":  true,
		},
	}
	ttm.CreateTemplate(dataProductTemplate)

	// Pipeline Template
	pipelineTemplate := &TwinTemplate{
		ID:          "template-pipeline",
		Name:        "Pipeline Template",
		Description: "Standard template for pipeline digital twins",
		Type:        "pipeline",
		Configuration: TwinConfiguration{
			ReplicationLevel: 0.9,
			SimulationMode:   "full",
			DataGeneration: DataGenerationConfig{
				Strategy:    "replay",
				Volume:      50000,
				Distribution: "exponential",
			},
		},
		Metadata: map[string]interface{}{
			"category": "pipeline",
			"default":  true,
		},
	}
	ttm.CreateTemplate(pipelineTemplate)
}

func mergeMetadata(meta1, meta2 map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range meta1 {
		result[k] = v
	}
	for k, v := range meta2 {
		result[k] = v
	}
	return result
}

