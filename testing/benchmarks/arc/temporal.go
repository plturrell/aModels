package arc

// Temporal pattern detection and sequence reasoning for ARC
// Handles patterns that evolve over multiple steps

// TemporalSequence represents a sequence of grid states
type TemporalSequence struct {
	States [][]int
}

// TemporalPattern represents a detected pattern in a sequence
type TemporalPattern struct {
	Type      string  // "linear", "cyclic", "growth", "decay", "oscillation"
	Period    int     // For cyclic patterns
	Direction Coord   // For directional patterns
	Rate      float64 // For growth/decay patterns
}

// DetectTemporalPattern analyzes a sequence to find patterns
func DetectTemporalPattern(sequence [][][]int) *TemporalPattern {
	if len(sequence) < 2 {
		return nil
	}

	// Check for cyclic patterns
	if period := detectCyclicPeriod(sequence); period > 0 {
		return &TemporalPattern{
			Type:   "cyclic",
			Period: period,
		}
	}

	// Check for linear growth/decay
	if rate := detectGrowthRate(sequence); rate != 0 {
		patternType := "growth"
		if rate < 0 {
			patternType = "decay"
		}
		return &TemporalPattern{
			Type: patternType,
			Rate: rate,
		}
	}

	// Check for directional movement
	if dir := detectMovementDirection(sequence); dir.Row != 0 || dir.Col != 0 {
		return &TemporalPattern{
			Type:      "linear",
			Direction: dir,
		}
	}

	return nil
}

func detectCyclicPeriod(sequence [][][]int) int {
	n := len(sequence)
	for period := 1; period <= n/2; period++ {
		isCyclic := true
		for i := 0; i < n-period; i++ {
			if !equalGrid(sequence[i], sequence[i+period]) {
				isCyclic = false
				break
			}
		}
		if isCyclic {
			return period
		}
	}
	return 0
}

func detectGrowthRate(sequence [][][]int) float64 {
	if len(sequence) < 2 {
		return 0
	}

	sizes := make([]int, len(sequence))
	for i, grid := range sequence {
		count := 0
		for _, row := range grid {
			for _, cell := range row {
				if cell != 0 {
					count++
				}
			}
		}
		sizes[i] = count
	}

	// Calculate average growth rate
	totalChange := 0
	for i := 1; i < len(sizes); i++ {
		totalChange += sizes[i] - sizes[i-1]
	}

	return float64(totalChange) / float64(len(sizes)-1)
}

func detectMovementDirection(sequence [][][]int) Coord {
	if len(sequence) < 2 {
		return Coord{0, 0}
	}

	// Find objects in first and last state
	objects1 := DetectObjects(sequence[0], 0)
	objects2 := DetectObjects(sequence[len(sequence)-1], 0)

	if len(objects1) == 0 || len(objects2) == 0 {
		return Coord{0, 0}
	}

	// Calculate average movement
	totalDR, totalDC := 0, 0
	matches := 0

	for _, obj1 := range objects1 {
		for _, obj2 := range objects2 {
			if obj1.Color == obj2.Color && obj1.Size == obj2.Size {
				totalDR += obj2.Centroid.Row - obj1.Centroid.Row
				totalDC += obj2.Centroid.Col - obj1.Centroid.Col
				matches++
			}
		}
	}

	if matches == 0 {
		return Coord{0, 0}
	}

	return Coord{
		Row: totalDR / matches,
		Col: totalDC / matches,
	}
}

// TemporalPredictOp predicts the next state in a sequence
type TemporalPredictOp struct {
	Sequence [][][]int
}

func (op *TemporalPredictOp) Apply(grid [][]int) [][]int {
	pattern := DetectTemporalPattern(op.Sequence)
	if pattern == nil {
		return grid
	}

	switch pattern.Type {
	case "cyclic":
		// Return the next state in the cycle
		idx := len(op.Sequence) % pattern.Period
		return CloneGrid(op.Sequence[idx])

	case "linear":
		// Apply movement
		return applyMovement(grid, pattern.Direction)

	case "growth":
		// Expand objects
		return expandObjects(grid, int(pattern.Rate))

	case "decay":
		// Shrink objects
		return shrinkObjects(grid, int(-pattern.Rate))
	}

	return grid
}

func (op *TemporalPredictOp) Name() string {
	return "temporal_predict"
}

func applyMovement(grid [][]int, dir Coord) [][]int {
	h, w := len(grid), len(grid[0])
	result := make([][]int, h)
	for i := range result {
		result[i] = make([]int, w)
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			newI := i + dir.Row
			newJ := j + dir.Col
			if newI >= 0 && newI < h && newJ >= 0 && newJ < w {
				result[newI][newJ] = grid[i][j]
			}
		}
	}

	return result
}

func expandObjects(grid [][]int, amount int) [][]int {
	result := CloneGrid(grid)
	h, w := len(result), len(result[0])

	for step := 0; step < amount; step++ {
		snapshot := CloneGrid(result)
		for i := 0; i < h; i++ {
			for j := 0; j < w; j++ {
				if snapshot[i][j] != 0 {
					// Expand to neighbors
					dirs := []Coord{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
					for _, d := range dirs {
						ni, nj := i+d.Row, j+d.Col
						if ni >= 0 && ni < h && nj >= 0 && nj < w && result[ni][nj] == 0 {
							result[ni][nj] = snapshot[i][j]
						}
					}
				}
			}
		}
	}

	return result
}

func shrinkObjects(grid [][]int, amount int) [][]int {
	result := CloneGrid(grid)
	h, w := len(result), len(result[0])

	for step := 0; step < amount; step++ {
		snapshot := CloneGrid(result)
		for i := 0; i < h; i++ {
			for j := 0; j < w; j++ {
				if snapshot[i][j] != 0 {
					// Check if on border of object
					dirs := []Coord{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
					hasBg := false
					for _, d := range dirs {
						ni, nj := i+d.Row, j+d.Col
						if ni < 0 || ni >= h || nj < 0 || nj >= w || snapshot[ni][nj] == 0 {
							hasBg = true
							break
						}
					}
					if hasBg {
						result[i][j] = 0
					}
				}
			}
		}
	}

	return result
}

// AnimationOp generates intermediate states between two grids
type AnimationOp struct {
	StartGrid [][]int
	EndGrid   [][]int
	Steps     int
}

func (op *AnimationOp) Apply(grid [][]int) [][]int {
	// Generate interpolated state
	// This is a simplified version - real implementation would be more sophisticated
	return op.EndGrid
}

func (op *AnimationOp) Name() string {
	return "animation"
}
