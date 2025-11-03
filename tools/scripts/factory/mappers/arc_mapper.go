package mappers

import (
	"ai_benchmarks/benchmarks/arc"
	"ai_benchmarks/scripts/datagen"
	"ai_benchmarks/scripts/factory"
	"fmt"
	"strconv"
)

// ARCMapper transforms data into abstract reasoning grid tasks.
// Uses the ARC benchmark's FactoryBridge for full DSL integration.
type ARCMapper struct {
	generator *arc.FactoryTaskGenerator
}

func NewARCMapper(seed int64) *ARCMapper {
	return &ARCMapper{
		generator: arc.NewFactoryTaskGenerator("default", seed),
	}
}

// NewARCMapperWithModel creates a mapper with model-specific configuration.
func NewARCMapperWithModel(modelName string, seed int64) *ARCMapper {
	return &ARCMapper{
		generator: arc.NewFactoryTaskGenerator(modelName, seed),
	}
}

func (m *ARCMapper) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for ARCMapper")
	}

	inputType := row["type"]
	switch inputType {
	case "image":
		return m.mapFromImage(row)
	case "grid":
		return m.mapFromGrid(row)
	case "pattern":
		return m.mapFromPattern(row)
	default:
		return m.mapFromPattern(row)
	}
}

func (m *ARCMapper) mapFromImage(row map[string]string) ([]factory.BenchmarkTask, error) {
	imagePath := row["image_path"]
	if imagePath == "" {
		return nil, fmt.Errorf("no image_path provided")
	}

	grid, err := datagen.ProcessImage(imagePath)
	if err != nil {
		return nil, err
	}

	rule := row["rule"]
	if rule == "" {
		rule = "identity"
	}

	// Use the ARC bridge to generate task with full DSL
	task := m.generator.GenerateFromImage(grid, rule)

	return []factory.BenchmarkTask{task}, nil
}

func (m *ARCMapper) mapFromGrid(row map[string]string) ([]factory.BenchmarkTask, error) {
	inputGrid := parseGridString(row["input_grid"])
	outputGrid := parseGridString(row["output_grid"])

	if inputGrid == nil || outputGrid == nil {
		return nil, fmt.Errorf("invalid grid data")
	}

	// Create task using ARC's native types
	task := arc.Task{
		Train: []arc.Pair{{In: inputGrid, Out: outputGrid}},
		Test:  []arc.Pair{{In: inputGrid, Out: outputGrid}},
	}

	return []factory.BenchmarkTask{task}, nil
}

func (m *ARCMapper) mapFromPattern(row map[string]string) ([]factory.BenchmarkTask, error) {
	pattern := row["pattern"]
	if pattern == "" {
		pattern = "identity"
	}

	size := 10
	if s := row["size"]; s != "" {
		if parsed, err := strconv.Atoi(s); err == nil {
			size = parsed
		}
	}

	// Use the ARC bridge to generate task with full DSL
	task := m.generator.GenerateFromPattern(pattern, size)

	return []factory.BenchmarkTask{task}, nil
}

// parseGridString is a placeholder for JSON grid parsing
func parseGridString(s string) [][]int {
	// TODO: Implement proper JSON parsing
	return nil
}
