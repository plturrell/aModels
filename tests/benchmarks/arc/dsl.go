package arc

// DSL - Domain Specific Language for ARC transformations
// Implements richer operations beyond simple geometric transforms

import (
	"math"
)

// Operation represents a transformation operation in the DSL
type Operation interface {
	Apply(grid [][]int) [][]int
	Name() string
}

// --- Counting Operations ---

type CountColorOp struct {
	Color int
}

func (op CountColorOp) Apply(grid [][]int) [][]int {
	count := 0
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == op.Color {
				count++
			}
		}
	}
	// Encode count as a simple grid representation
	return createCountGrid(count)
}

func (op CountColorOp) Name() string {
	return "count_color"
}

func createCountGrid(count int) [][]int {
	// Simple encoding: create a 1xN grid where N = count
	if count == 0 {
		return [][]int{{0}}
	}
	row := make([]int, count)
	for i := range row {
		row[i] = 1
	}
	return [][]int{row}
}

// --- Object-based Operations ---

type ExtractLargestObjectOp struct {
	BackgroundColor int
}

func (op ExtractLargestObjectOp) Apply(grid [][]int) [][]int {
	objects := DetectObjects(grid, op.BackgroundColor)
	if len(objects) == 0 {
		return grid
	}

	largest := objects[0]
	for _, obj := range objects[1:] {
		if obj.Size > largest.Size {
			largest = obj
		}
	}

	return extractObjectToGrid(largest, grid)
}

func (op ExtractLargestObjectOp) Name() string {
	return "extract_largest"
}

func extractObjectToGrid(obj Object, original [][]int) [][]int {
	h := obj.Bounds.MaxRow - obj.Bounds.MinRow + 1
	w := obj.Bounds.MaxCol - obj.Bounds.MinCol + 1
	grid := make([][]int, h)
	for i := range grid {
		grid[i] = make([]int, w)
	}

	pixelSet := make(map[Coord]bool)
	for _, p := range obj.Pixels {
		pixelSet[p] = true
	}

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			origR := i + obj.Bounds.MinRow
			origC := j + obj.Bounds.MinCol
			if pixelSet[Coord{origR, origC}] {
				grid[i][j] = original[origR][origC]
			}
		}
	}
	return grid
}

type ReplicateObjectOp struct {
	Times           int
	Direction       string // "horizontal", "vertical", "diagonal"
	BackgroundColor int
}

func (op ReplicateObjectOp) Apply(grid [][]int) [][]int {
	objects := DetectObjects(grid, op.BackgroundColor)
	if len(objects) == 0 {
		return grid
	}

	// Replicate the first object
	obj := objects[0]
	h, w := len(grid), len(grid[0])

	switch op.Direction {
	case "horizontal":
		newW := w * op.Times
		newGrid := make([][]int, h)
		for i := range newGrid {
			newGrid[i] = make([]int, newW)
		}
		for t := 0; t < op.Times; t++ {
			offset := t * w
			for _, p := range obj.Pixels {
				if p.Col+offset < newW {
					newGrid[p.Row][p.Col+offset] = grid[p.Row][p.Col]
				}
			}
		}
		return newGrid

	case "vertical":
		newH := h * op.Times
		newGrid := make([][]int, newH)
		for i := range newGrid {
			newGrid[i] = make([]int, w)
		}
		for t := 0; t < op.Times; t++ {
			offset := t * h
			for _, p := range obj.Pixels {
				if p.Row+offset < newH {
					newGrid[p.Row+offset][p.Col] = grid[p.Row][p.Col]
				}
			}
		}
		return newGrid
	}

	return grid
}

func (op ReplicateObjectOp) Name() string {
	return "replicate_object"
}

// --- Relational Operations ---

type AlignObjectsOp struct {
	Alignment       string // "top", "bottom", "left", "right", "center"
	BackgroundColor int
}

func (op AlignObjectsOp) Apply(grid [][]int) [][]int {
	objects := DetectObjects(grid, op.BackgroundColor)
	if len(objects) < 2 {
		return grid
	}

	h, w := len(grid), len(grid[0])
	newGrid := make([][]int, h)
	for i := range newGrid {
		newGrid[i] = make([]int, w)
		for j := range newGrid[i] {
			newGrid[i][j] = op.BackgroundColor
		}
	}

	// Align all objects to the first object's position
	ref := objects[0]
	var refPos int
	switch op.Alignment {
	case "top":
		refPos = ref.Bounds.MinRow
	case "bottom":
		refPos = ref.Bounds.MaxRow
	case "left":
		refPos = ref.Bounds.MinCol
	case "right":
		refPos = ref.Bounds.MaxCol
	case "center":
		refPos = ref.Centroid.Row
	}

	for _, obj := range objects {
		var offset int
		switch op.Alignment {
		case "top":
			offset = refPos - obj.Bounds.MinRow
		case "bottom":
			offset = refPos - obj.Bounds.MaxRow
		case "left":
			offset = refPos - obj.Bounds.MinCol
		case "right":
			offset = refPos - obj.Bounds.MaxCol
		case "center":
			offset = refPos - obj.Centroid.Row
		}

		for _, p := range obj.Pixels {
			var newR, newC int
			if op.Alignment == "top" || op.Alignment == "bottom" || op.Alignment == "center" {
				newR = p.Row + offset
				newC = p.Col
			} else {
				newR = p.Row
				newC = p.Col + offset
			}

			if newR >= 0 && newR < h && newC >= 0 && newC < w {
				newGrid[newR][newC] = obj.Color
			}
		}
	}

	return newGrid
}

func (op AlignObjectsOp) Name() string {
	return "align_objects"
}

// --- Pattern Operations ---

type DetectPatternOp struct {
	PatternType string // "repeating", "symmetric", "grid"
}

func (op DetectPatternOp) Apply(grid [][]int) [][]int {
	switch op.PatternType {
	case "repeating":
		return detectRepeatingPattern(grid)
	case "symmetric":
		return detectSymmetricPattern(grid)
	}
	return grid
}

func (op DetectPatternOp) Name() string {
	return "detect_pattern"
}

func detectRepeatingPattern(grid [][]int) [][]int {
	h, w := len(grid), len(grid[0])
	if h == 0 || w == 0 {
		return grid
	}

	// Try to find horizontal repetition
	for period := 1; period <= w/2; period++ {
		if w%period == 0 {
			isRepeating := true
			for i := 0; i < h && isRepeating; i++ {
				for j := period; j < w && isRepeating; j++ {
					if grid[i][j] != grid[i][j%period] {
						isRepeating = false
					}
				}
			}
			if isRepeating {
				// Extract the pattern
				pattern := make([][]int, h)
				for i := range pattern {
					pattern[i] = make([]int, period)
					copy(pattern[i], grid[i][:period])
				}
				return pattern
			}
		}
	}

	return grid
}

func detectSymmetricPattern(grid [][]int) [][]int {
	h, w := len(grid), len(grid[0])
	if h == 0 || w == 0 {
		return grid
	}

	// Check for vertical symmetry
	isSymmetric := true
	for i := 0; i < h && isSymmetric; i++ {
		for j := 0; j < w/2 && isSymmetric; j++ {
			if grid[i][j] != grid[i][w-1-j] {
				isSymmetric = false
			}
		}
	}

	if isSymmetric {
		// Return left half
		half := make([][]int, h)
		for i := range half {
			half[i] = make([]int, (w+1)/2)
			copy(half[i], grid[i][:(w+1)/2])
		}
		return half
	}

	return grid
}

// --- Goal Inference Operations ---

type InferGoalOp struct {
	TrainPairs []Pair
}

func (op InferGoalOp) Apply(grid [][]int) [][]int {
	// Analyze training pairs to infer the transformation goal
	if len(op.TrainPairs) == 0 {
		return grid
	}

	// Check if goal is to extract objects of a certain size
	sizePattern := analyzeObjectSizePattern(op.TrainPairs)
	if sizePattern > 0 {
		objects := DetectObjects(grid, 0)
		filtered := FilterObjectsBySize(objects, sizePattern-2, sizePattern+2)
		if len(filtered) > 0 {
			return extractObjectToGrid(filtered[0], grid)
		}
	}

	// Check if goal is to count objects
	if isCountingTask(op.TrainPairs) {
		objects := DetectObjects(grid, 0)
		count := len(objects)
		return createCountGrid(count)
	}

	return grid
}

func (op InferGoalOp) Name() string {
	return "infer_goal"
}

func analyzeObjectSizePattern(pairs []Pair) int {
	sizes := []int{}
	for _, p := range pairs {
		objects := DetectObjects(p.Out, 0)
		for _, obj := range objects {
			sizes = append(sizes, obj.Size)
		}
	}

	if len(sizes) == 0 {
		return 0
	}

	// Return most common size
	sizeCount := make(map[int]int)
	for _, s := range sizes {
		sizeCount[s]++
	}

	maxCount := 0
	commonSize := 0
	for size, count := range sizeCount {
		if count > maxCount {
			maxCount = count
			commonSize = size
		}
	}

	return commonSize
}

func isCountingTask(pairs []Pair) bool {
	for _, p := range pairs {
		inObjects := DetectObjects(p.In, 0)
		outH, outW := len(p.Out), 0
		if outH > 0 {
			outW = len(p.Out[0])
		}

		// Check if output is a simple 1D representation
		if outH == 1 && outW == len(inObjects) {
			return true
		}
	}
	return false
}

// --- Composition Operations ---

type CompositeOp struct {
	Operations []Operation
}

func (op CompositeOp) Apply(grid [][]int) [][]int {
	result := grid
	for _, subOp := range op.Operations {
		result = subOp.Apply(result)
	}
	return result
}

func (op CompositeOp) Name() string {
	return "composite"
}

// --- Utility Functions ---

func CloneGrid(grid [][]int) [][]int {
	if len(grid) == 0 {
		return nil
	}
	clone := make([][]int, len(grid))
	for i := range grid {
		clone[i] = make([]int, len(grid[i]))
		copy(clone[i], grid[i])
	}
	return clone
}

func GridDistance(a, b [][]int) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	dist := 0.0
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return math.Inf(1)
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				dist += 1.0
			}
		}
	}
	return dist
}
