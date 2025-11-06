package graph

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

// MurexCatalogPopulator populates the catalog with Murex-specific data elements.
type MurexCatalogPopulator struct {
	extractor     *MurexTerminologyExtractor
	registry      *iso11179.MetadataRegistry
	logger        *log.Logger
}

// NewMurexCatalogPopulator creates a new Murex catalog populator.
func NewMurexCatalogPopulator(
	extractor *MurexTerminologyExtractor,
	registry *iso11179.MetadataRegistry,
	logger *log.Logger,
) *MurexCatalogPopulator {
	return &MurexCatalogPopulator{
		extractor: extractor,
		registry:  registry,
		logger:    logger,
	}
}

// PopulateFromTerminology populates the catalog from extracted terminology.
func (mcp *MurexCatalogPopulator) PopulateFromTerminology(ctx context.Context) error {
	if mcp.logger != nil {
		mcp.logger.Printf("Populating catalog from Murex terminology")
	}

	terminology := mcp.extractor.GetTerminology()

	// Register domain-specific data elements
	for domain, examples := range terminology.Domains {
		for _, example := range examples {
			element := iso11179.NewDataElement(
				fmt.Sprintf("murex:%s:%s", domain, example.Text),
				example.Text,
				fmt.Sprintf("Murex %s terminology: %s", domain, example.Text),
				"terminology",
				fmt.Sprintf("http://amodels.org/catalog/murex/%s/%s", domain, example.Text),
			)
			element.AddMetadata("source", "murex")
			element.AddMetadata("domain", domain)
			element.AddMetadata("confidence", example.Confidence)
			element.AddMetadata("timestamp", example.Timestamp.Format(time.RFC3339))

			if err := mcp.registry.RegisterDataElement(element); err != nil {
				if mcp.logger != nil {
					mcp.logger.Printf("Warning: Failed to register data element %s: %v", element.ID, err)
				}
			}
		}
	}

	// Register role-specific data elements
	for role, examples := range terminology.Roles {
		for _, example := range examples {
			element := iso11179.NewDataElement(
				fmt.Sprintf("murex:role:%s:%s", role, example.Text),
				example.Text,
				fmt.Sprintf("Murex field role: %s (role: %s)", example.Text, role),
				"field_role",
				fmt.Sprintf("http://amodels.org/catalog/murex/roles/%s/%s", role, example.Text),
			)
			element.AddMetadata("source", "murex")
			element.AddMetadata("role", role)
			element.AddMetadata("business_role", role)
			element.AddMetadata("confidence", example.Confidence)

			if err := mcp.registry.RegisterDataElement(element); err != nil {
				if mcp.logger != nil {
					mcp.logger.Printf("Warning: Failed to register role element %s: %v", element.ID, err)
				}
			}
		}
	}

	if mcp.logger != nil {
		mcp.logger.Printf("Populated catalog with Murex terminology")
	}

	return nil
}

// PopulateFromTrainingData populates the catalog from training data schemas.
func (mcp *MurexCatalogPopulator) PopulateFromTrainingData(ctx context.Context) error {
	if mcp.logger != nil {
		mcp.logger.Printf("Populating catalog from Murex training data")
	}

	trainingData := mcp.extractor.GetTrainingData()

	// Register schema examples as data elements
	for _, schemaExample := range trainingData.SchemaExamples {
		// Register table/entity
		tableElement := iso11179.NewDataElement(
			fmt.Sprintf("murex:table:%s", schemaExample.TableName),
			schemaExample.TableName,
			schemaExample.Description,
			"table",
			fmt.Sprintf("http://amodels.org/catalog/murex/tables/%s", schemaExample.TableName),
		)
		tableElement.AddMetadata("source", "murex")
		tableElement.AddMetadata("source_system", "murex")
		tableElement.AddMetadata("table_name", schemaExample.TableName)

		if err := mcp.registry.RegisterDataElement(tableElement); err != nil {
			if mcp.logger != nil {
				mcp.logger.Printf("Warning: Failed to register table element %s: %v", tableElement.ID, err)
			}
		}

		// Register columns/fields
		for _, column := range schemaExample.Columns {
			columnElement := iso11179.NewDataElement(
				fmt.Sprintf("murex:column:%s:%s", schemaExample.TableName, column.Name),
				column.Name,
				column.Description,
				"column",
				fmt.Sprintf("http://amodels.org/catalog/murex/columns/%s/%s", schemaExample.TableName, column.Name),
			)
			columnElement.AddMetadata("source", "murex")
			columnElement.AddMetadata("table", schemaExample.TableName)
			columnElement.AddMetadata("data_type", column.Type)
			columnElement.AddMetadata("nullable", column.Nullable)
			if len(column.Examples) > 0 {
				columnElement.AddMetadata("example_value", fmt.Sprintf("%v", column.Examples[0]))
			}

			if err := mcp.registry.RegisterDataElement(columnElement); err != nil {
				if mcp.logger != nil {
					mcp.logger.Printf("Warning: Failed to register column element %s: %v", columnElement.ID, err)
				}
			}
		}
	}

	// Register field examples
	for _, fieldExample := range trainingData.FieldExamples {
		element := iso11179.NewDataElement(
			fmt.Sprintf("murex:field:%s", fieldExample.FieldName),
			fieldExample.FieldName,
			fieldExample.Description,
			"field",
			fmt.Sprintf("http://amodels.org/catalog/murex/fields/%s", fieldExample.FieldName),
		)
		element.AddMetadata("source", "murex")
		element.AddMetadata("domain", fieldExample.Domain)
		element.AddMetadata("role", fieldExample.Role)
		element.AddMetadata("pattern", fieldExample.Pattern)
		element.AddMetadata("data_type", fieldExample.FieldType)

		if err := mcp.registry.RegisterDataElement(element); err != nil {
			if mcp.logger != nil {
				mcp.logger.Printf("Warning: Failed to register field element %s: %v", element.ID, err)
			}
		}
	}

	if mcp.logger != nil {
		mcp.logger.Printf("Populated catalog with %d schema examples and %d field examples",
			len(trainingData.SchemaExamples), len(trainingData.FieldExamples))
	}

	return nil
}

// PopulateAll populates the catalog with all extracted information.
func (mcp *MurexCatalogPopulator) PopulateAll(ctx context.Context) error {
	if err := mcp.PopulateFromTerminology(ctx); err != nil {
		return fmt.Errorf("failed to populate from terminology: %w", err)
	}

	if err := mcp.PopulateFromTrainingData(ctx); err != nil {
		return fmt.Errorf("failed to populate from training data: %w", err)
	}

	return nil
}

