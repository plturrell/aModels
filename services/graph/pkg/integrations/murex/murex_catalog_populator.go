package murex

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/graph/pkg/clients/catalog"
)

// MurexCatalogPopulator populates the catalog with Murex-specific data elements.
// Now uses HTTP client instead of direct registry access for microservice architecture.
type MurexCatalogPopulator struct {
	extractor     *MurexTerminologyExtractor
	registry      *iso11179.MetadataRegistry // Kept for backward compatibility, but deprecated
	catalogClient *catalog.CatalogClient     // HTTP client for catalog service
	logger        *log.Logger
	useHTTP       bool // Flag to choose between HTTP and direct registry
}

// NewMurexCatalogPopulator creates a new Murex catalog populator.
// If catalogClient is provided, uses HTTP. Otherwise falls back to direct registry (deprecated).
func NewMurexCatalogPopulator(
	extractor *MurexTerminologyExtractor,
	registry *iso11179.MetadataRegistry,
	catalogClient *catalog.CatalogClient,
	logger *log.Logger,
) *MurexCatalogPopulator {
	useHTTP := catalogClient != nil && catalogClient.enabled
	return &MurexCatalogPopulator{
		extractor:     extractor,
		registry:      registry,
		catalogClient: catalogClient,
		logger:        logger,
		useHTTP:       useHTTP,
	}
}

// PopulateFromTerminology populates the catalog from extracted terminology.
func (mcp *MurexCatalogPopulator) PopulateFromTerminology(ctx context.Context) error {
	method := "direct registry"
	if mcp.useHTTP {
		method = "HTTP"
	}
	if mcp.logger != nil {
		mcp.logger.Printf("Populating catalog from Murex terminology (using %s)", method)
	}

	terminology := mcp.extractor.GetTerminology()

	var elements []DataElementRequest

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
			element.AddMetadata("confidence", fmt.Sprintf("%f", example.Confidence))
			element.AddMetadata("timestamp", example.Timestamp.Format(time.RFC3339))

			if mcp.useHTTP {
				elements = append(elements, ConvertISO11179ToRequest(element))
			} else {
				// Fallback to direct registry (deprecated)
				if mcp.registry != nil {
					mcp.registry.RegisterDataElement(element)
				}
				if mcp.logger != nil {
					mcp.logger.Printf("Registered data element %s (direct)", element.Identifier)
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
			element.AddMetadata("confidence", fmt.Sprintf("%f", example.Confidence))

			if mcp.useHTTP {
				elements = append(elements, ConvertISO11179ToRequest(element))
			} else {
				// Fallback to direct registry (deprecated)
				if mcp.registry != nil {
					mcp.registry.RegisterDataElement(element)
				}
				if mcp.logger != nil {
					mcp.logger.Printf("Registered role element %s (direct)", element.Identifier)
				}
			}
		}
	}

	// Register via HTTP if using HTTP client
	if mcp.useHTTP && len(elements) > 0 {
		if err := mcp.catalogClient.RegisterDataElementsBulk(ctx, elements); err != nil {
			// Log error but don't fail - catalog registration is best effort
			if mcp.logger != nil {
				mcp.logger.Printf("Warning: Failed to register terminology elements via HTTP: %v", err)
			}
			// Continue - don't fail the entire operation
		} else {
			if mcp.logger != nil {
				mcp.logger.Printf("Registered %d terminology elements via HTTP", len(elements))
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
	method := "direct registry"
	if mcp.useHTTP {
		method = "HTTP"
	}
	if mcp.logger != nil {
		mcp.logger.Printf("Populating catalog from Murex training data (using %s)", method)
	}

	trainingData := mcp.extractor.GetTrainingData()
	var elements []DataElementRequest

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

		if mcp.useHTTP {
			elements = append(elements, ConvertISO11179ToRequest(tableElement))
		} else {
			if mcp.registry != nil {
				mcp.registry.RegisterDataElement(tableElement)
			}
			if mcp.logger != nil {
				mcp.logger.Printf("Registered table element %s (direct)", tableElement.Identifier)
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
			columnElement.AddMetadata("nullable", fmt.Sprintf("%v", column.Nullable))
			if len(column.Examples) > 0 {
				columnElement.AddMetadata("example_value", fmt.Sprintf("%v", column.Examples[0]))
			}

			if mcp.useHTTP {
				elements = append(elements, ConvertISO11179ToRequest(columnElement))
			} else {
				if mcp.registry != nil {
					mcp.registry.RegisterDataElement(columnElement)
				}
				if mcp.logger != nil {
					mcp.logger.Printf("Registered column element %s (direct)", columnElement.Identifier)
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

		if mcp.useHTTP {
			elements = append(elements, ConvertISO11179ToRequest(element))
		} else {
			if mcp.registry != nil {
				mcp.registry.RegisterDataElement(element)
			}
			if mcp.logger != nil {
				mcp.logger.Printf("Registered field element %s (direct)", element.Identifier)
			}
		}
	}

	// Register via HTTP if using HTTP client
	if mcp.useHTTP && len(elements) > 0 {
		if err := mcp.catalogClient.RegisterDataElementsBulk(ctx, elements); err != nil {
			// Log error but don't fail - catalog registration is best effort
			if mcp.logger != nil {
				mcp.logger.Printf("Warning: Failed to register training data elements via HTTP: %v", err)
			}
			// Continue - don't fail the entire operation
		} else {
			if mcp.logger != nil {
				mcp.logger.Printf("Registered %d training data elements via HTTP", len(elements))
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

