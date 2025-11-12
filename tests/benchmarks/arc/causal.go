package arc

// Causal reasoning operations for ARC
// Implements cause-effect relationships and conditional transformations

// CausalRule represents a cause-effect transformation rule
type CausalRule struct {
	Condition func([][]int) bool
	Effect    Operation
}

// CausalOp applies transformations based on grid conditions
type CausalOp struct {
	Rules []CausalRule
}

func (op *CausalOp) Apply(grid [][]int) [][]int {
	for _, rule := range op.Rules {
		if rule.Condition(grid) {
			return rule.Effect.Apply(grid)
		}
	}
	return grid
}

func (op *CausalOp) Name() string {
	return "causal"
}

// Predefined causal conditions

// HasColorCondition checks if a specific color exists in the grid
func HasColorCondition(color int) func([][]int) bool {
	return func(grid [][]int) bool {
		for i := range grid {
			for j := range grid[i] {
				if grid[i][j] == color {
					return true
				}
			}
		}
		return false
	}
}

// ObjectCountCondition checks if object count meets criteria
func ObjectCountCondition(bgColor, minCount, maxCount int) func([][]int) bool {
	return func(grid [][]int) bool {
		objects := DetectObjects(grid, bgColor)
		count := len(objects)
		return count >= minCount && count <= maxCount
	}
}

// SymmetryCondition checks if grid has specific symmetry
func SymmetryCondition(symType SymmetryType) func([][]int) bool {
	return func(grid [][]int) bool {
		objects := DetectObjects(grid, 0)
		for _, obj := range objects {
			if obj.Symmetry == symType {
				return true
			}
		}
		return false
	}
}

// SizeCondition checks if grid meets size criteria
func SizeCondition(minH, minW int) func([][]int) bool {
	return func(grid [][]int) bool {
		return len(grid) >= minH && len(grid[0]) >= minW
	}
}

// ConditionalColorOp changes colors based on conditions
type ConditionalColorOp struct {
	SourceColor int
	TargetColor int
	Condition   func(int, int, [][]int) bool // row, col, grid
}

func (op *ConditionalColorOp) Apply(grid [][]int) [][]int {
	result := CloneGrid(grid)
	for i := range result {
		for j := range result[i] {
			if result[i][j] == op.SourceColor && op.Condition(i, j, grid) {
				result[i][j] = op.TargetColor
			}
		}
	}
	return result
}

func (op *ConditionalColorOp) Name() string {
	return "conditional_color"
}

// Spatial conditions for ConditionalColorOp

// HasNeighborCondition checks if cell has neighbor of specific color
func HasNeighborCondition(color int) func(int, int, [][]int) bool {
	return func(row, col int, grid [][]int) bool {
		h, w := len(grid), len(grid[0])
		dirs := []Coord{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
		for _, d := range dirs {
			nr, nc := row+d.Row, col+d.Col
			if nr >= 0 && nr < h && nc >= 0 && nc < w {
				if grid[nr][nc] == color {
					return true
				}
			}
		}
		return false
	}
}

// IsBorderCondition checks if cell is on grid border
func IsBorderCondition() func(int, int, [][]int) bool {
	return func(row, col int, grid [][]int) bool {
		h, w := len(grid), len(grid[0])
		return row == 0 || row == h-1 || col == 0 || col == w-1
	}
}

// IsIsolatedCondition checks if cell has no neighbors of same color
func IsIsolatedCondition() func(int, int, [][]int) bool {
	return func(row, col int, grid [][]int) bool {
		h, w := len(grid), len(grid[0])
		cellColor := grid[row][col]
		dirs := []Coord{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
		for _, d := range dirs {
			nr, nc := row+d.Row, col+d.Col
			if nr >= 0 && nr < h && nc >= 0 && nc < w {
				if grid[nr][nc] == cellColor {
					return false
				}
			}
		}
		return true
	}
}

// PropagationOp spreads a color based on causal rules
type PropagationOp struct {
	SourceColor int
	TargetColor int
	MaxSteps    int
	PropagateIf func(int, int, [][]int) bool
}

func (op *PropagationOp) Apply(grid [][]int) [][]int {
	result := CloneGrid(grid)
	h, w := len(result), len(result[0])

	for step := 0; step < op.MaxSteps; step++ {
		changed := false
		snapshot := CloneGrid(result)

		for i := 0; i < h; i++ {
			for j := 0; j < w; j++ {
				if snapshot[i][j] == op.SourceColor && op.PropagateIf(i, j, snapshot) {
					result[i][j] = op.TargetColor
					changed = true
				}
			}
		}

		if !changed {
			break
		}
	}

	return result
}

func (op *PropagationOp) Name() string {
	return "propagation"
}

// ChainReactionOp applies cascading transformations
type ChainReactionOp struct {
	Triggers []CausalRule
}

func (op *ChainReactionOp) Apply(grid [][]int) [][]int {
	result := grid
	maxIterations := 10

	for iter := 0; iter < maxIterations; iter++ {
		changed := false
		for _, trigger := range op.Triggers {
			if trigger.Condition(result) {
				newResult := trigger.Effect.Apply(result)
				if !equalGrid(newResult, result) {
					result = newResult
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}

	return result
}

func (op *ChainReactionOp) Name() string {
	return "chain_reaction"
}
