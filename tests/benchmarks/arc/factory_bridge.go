package arc

// FactoryBridge provides factory integration without circular dependencies.
// This allows the data factory to leverage ARC's sophisticated DSL and synthesis.

import (
	"math/rand"
)

// FactoryTaskGenerator creates ARC tasks from raw data using the full DSL.
type FactoryTaskGenerator struct {
	BackgroundColor int
	ModelConfig     ModelConfig
	UseSynthesis    bool
	Rand            *rand.Rand
}

// NewFactoryTaskGenerator creates a generator with model-aware configuration.
func NewFactoryTaskGenerator(modelName string, seed int64) *FactoryTaskGenerator {
	detectedModel := AutoDetectModel(modelName)
	modelCfg := GetModelConfig(detectedModel)

	return &FactoryTaskGenerator{
		BackgroundColor: 0,
		ModelConfig:     modelCfg,
		UseSynthesis:    modelCfg.EnableSynthesis,
		Rand:            rand.New(rand.NewSource(seed)),
	}
}

// GenerateFromPattern creates an ARC task using a pattern rule.
func (g *FactoryTaskGenerator) GenerateFromPattern(pattern string, size int) Task {
	inputGrid := generateSyntheticGrid(size, g.Rand)
	outputGrid := g.ApplyDSLOperation(inputGrid, pattern)

	// Generate test with variation
	testInput := generateSyntheticGrid(size, g.Rand)
	testOutput := g.ApplyDSLOperation(testInput, pattern)

	return Task{
		Train: []Pair{{In: inputGrid, Out: outputGrid}},
		Test:  []Pair{{In: testInput, Out: testOutput}},
	}
}

// GenerateFromImage creates an ARC task from an image grid.
func (g *FactoryTaskGenerator) GenerateFromImage(grid [][]int, rule string) Task {
	outputGrid := g.ApplyDSLOperation(grid, rule)

	return Task{
		Train: []Pair{{In: grid, Out: outputGrid}},
		Test:  []Pair{{In: grid, Out: outputGrid}},
	}
}

// GenerateWithSynthesis uses program synthesis to discover transformations.
func (g *FactoryTaskGenerator) GenerateWithSynthesis(trainPairs []Pair) Task {
	if !g.UseSynthesis || len(trainPairs) == 0 {
		// Fallback to simple task
		return Task{
			Train: trainPairs,
			Test:  trainPairs,
		}
	}

	ps := &ProgramSynthesis{
		MaxDepth:        g.ModelConfig.ReasoningDepth,
		MaxCandidates:   g.ModelConfig.MCTSRollouts / 10,
		BackgroundColor: g.BackgroundColor,
		Rand:            g.Rand,
	}

	program := ps.Synthesize(trainPairs)

	// Generate test by applying discovered program
	testInput := varyGrid(trainPairs[0].In, g.Rand)
	testOutput := ApplyProgram(testInput, program)

	return Task{
		Train: trainPairs,
		Test:  []Pair{{In: testInput, Out: testOutput}},
	}
}

// ApplyDSLOperation uses the full ARC DSL to transform grids.
func (g *FactoryTaskGenerator) ApplyDSLOperation(grid [][]int, operation string) [][]int {
	switch operation {
	case "identity":
		return CloneGrid(grid)

	// Object-based operations
	case "extract_largest_object":
		op := &ExtractLargestObjectOp{BackgroundColor: g.BackgroundColor}
		return op.Apply(grid)
	case "align_objects_top":
		op := &AlignObjectsOp{Alignment: "top", BackgroundColor: g.BackgroundColor}
		return op.Apply(grid)
	case "align_objects_left":
		op := &AlignObjectsOp{Alignment: "left", BackgroundColor: g.BackgroundColor}
		return op.Apply(grid)
	case "align_objects_center":
		op := &AlignObjectsOp{Alignment: "center", BackgroundColor: g.BackgroundColor}
		return op.Apply(grid)
	case "replicate_horizontal":
		op := &ReplicateObjectOp{Times: 2, Direction: "horizontal", BackgroundColor: g.BackgroundColor}
		return op.Apply(grid)
	case "replicate_vertical":
		op := &ReplicateObjectOp{Times: 2, Direction: "vertical", BackgroundColor: g.BackgroundColor}
		return op.Apply(grid)

	// Counting operations
	case "count_color_0":
		op := &CountColorOp{Color: 0}
		return op.Apply(grid)
	case "count_color_1":
		op := &CountColorOp{Color: 1}
		return op.Apply(grid)
	case "count_color_2":
		op := &CountColorOp{Color: 2}
		return op.Apply(grid)
	case "count_objects":
		objects := DetectObjects(grid, g.BackgroundColor)
		count := len(objects)
		return bridgeCreateCountGrid(count)

	// Pattern operations
	case "detect_pattern_repeating":
		op := &DetectPatternOp{PatternType: "repeating"}
		return op.Apply(grid)
	case "detect_pattern_symmetric":
		op := &DetectPatternOp{PatternType: "symmetric"}
		return op.Apply(grid)

	// Geometric operations (use existing functions from arc.go)
	case "flip_h":
		return flipH(grid)
	case "flip_v":
		return flipV(grid)
	case "rot90":
		return rot90(grid)
	case "rot180":
		return rot180(grid)
	case "rot270":
		return rot270(grid)

	// Causal operations
	case "conditional_color_border":
		op := &ConditionalColorOp{
			SourceColor: 1,
			TargetColor: 2,
			Condition:   IsBorderCondition(),
		}
		return op.Apply(grid)
	case "propagation":
		op := &PropagationOp{
			SourceColor: 0,
			TargetColor: 1,
			MaxSteps:    5,
			PropagateIf: HasNeighborCondition(1),
		}
		return op.Apply(grid)

	// Temporal operations
	case "expand_objects":
		return expandObjects(grid, 1)
	case "shrink_objects":
		return shrinkObjects(grid, 1)

	default:
		return CloneGrid(grid)
	}
}

// Helper functions

func generateSyntheticGrid(size int, rnd *rand.Rand) [][]int {
	grid := make([][]int, size)
	for i := range grid {
		grid[i] = make([]int, size)
		for j := range grid[i] {
			if rnd.Float64() < 0.3 {
				grid[i][j] = rnd.Intn(4)
			}
		}
	}
	return grid
}

func varyGrid(grid [][]int, rnd *rand.Rand) [][]int {
	varied := CloneGrid(grid)
	h, w := len(varied), len(varied[0])

	// Add small random variations
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if rnd.Float64() < 0.1 {
				varied[i][j] = rnd.Intn(4)
			}
		}
	}
	return varied
}

func bridgeCreateCountGrid(count int) [][]int {
	if count <= 0 {
		return [][]int{{0}}
	}
	row := make([]int, count)
	for i := range row {
		row[i] = 1
	}
	return [][]int{row}
}
