package arc

import "fmt"

// Hierarchical abstraction for ARC
// Implements multi-level reasoning from pixels → objects → groups → scenes

// AbstractionLevel represents different levels of reasoning
type AbstractionLevel int

const (
	PixelLevel AbstractionLevel = iota
	ObjectLevel
	GroupLevel
	SceneLevel
)

// HierarchicalRepresentation stores multi-level abstractions of a grid
type HierarchicalRepresentation struct {
	Pixels  [][]int           // Level 0: Raw pixels
	Objects []Object          // Level 1: Connected components
	Groups  []ObjectGroup     // Level 2: Related objects
	Scene   *SceneDescription // Level 3: High-level description
}

// ObjectGroup represents a collection of related objects
type ObjectGroup struct {
	Objects      []Object
	Relationship string // "aligned", "symmetric", "repeated", "nested"
	BoundingBox  Rectangle
	Centroid     Coord
}

// SceneDescription provides high-level semantic understanding
type SceneDescription struct {
	Layout     string   // "grid", "scattered", "centered", "border"
	Symmetries []string // "horizontal", "vertical", "rotational"
	Patterns   []string // "repeating", "alternating", "gradient"
	Complexity int      // 0-10 scale
	NumLayers  int      // Depth of nesting
}

// BuildHierarchy creates a hierarchical representation from a grid
func BuildHierarchy(grid [][]int, bgColor int) *HierarchicalRepresentation {
	hr := &HierarchicalRepresentation{
		Pixels: grid,
	}

	// Level 1: Detect objects
	hr.Objects = DetectObjects(grid, bgColor)

	// Level 2: Group related objects
	hr.Groups = detectObjectGroups(hr.Objects)

	// Level 3: Analyze scene
	hr.Scene = analyzeScene(grid, hr.Objects, hr.Groups)

	return hr
}

func detectObjectGroups(objects []Object) []ObjectGroup {
	if len(objects) < 2 {
		return nil
	}

	groups := []ObjectGroup{}

	// Group by alignment
	alignedGroups := groupByAlignment(objects)
	groups = append(groups, alignedGroups...)

	// Group by symmetry
	symmetricGroups := groupBySymmetry(objects)
	groups = append(groups, symmetricGroups...)

	// Group by repetition
	repeatedGroups := groupByRepetition(objects)
	groups = append(groups, repeatedGroups...)

	return groups
}

func groupByAlignment(objects []Object) []ObjectGroup {
	groups := []ObjectGroup{}

	// Horizontal alignment
	rowGroups := make(map[int][]Object)
	for _, obj := range objects {
		rowGroups[obj.Centroid.Row] = append(rowGroups[obj.Centroid.Row], obj)
	}
	for _, objs := range rowGroups {
		if len(objs) >= 2 {
			groups = append(groups, ObjectGroup{
				Objects:      objs,
				Relationship: "aligned_horizontal",
			})
		}
	}

	// Vertical alignment
	colGroups := make(map[int][]Object)
	for _, obj := range objects {
		colGroups[obj.Centroid.Col] = append(colGroups[obj.Centroid.Col], obj)
	}
	for _, objs := range colGroups {
		if len(objs) >= 2 {
			groups = append(groups, ObjectGroup{
				Objects:      objs,
				Relationship: "aligned_vertical",
			})
		}
	}

	return groups
}

func groupBySymmetry(objects []Object) []ObjectGroup {
	groups := []ObjectGroup{}

	// Find pairs of objects that are symmetric
	for i := 0; i < len(objects); i++ {
		for j := i + 1; j < len(objects); j++ {
			if areSymmetric(objects[i], objects[j]) {
				groups = append(groups, ObjectGroup{
					Objects:      []Object{objects[i], objects[j]},
					Relationship: "symmetric",
				})
			}
		}
	}

	return groups
}

func groupByRepetition(objects []Object) []ObjectGroup {
	groups := []ObjectGroup{}

	// Group objects with same size and color
	sizeColorGroups := make(map[string][]Object)
	for _, obj := range objects {
		key := fmt.Sprintf("%d_%d", obj.Size, obj.Color)
		sizeColorGroups[key] = append(sizeColorGroups[key], obj)
	}

	for _, objs := range sizeColorGroups {
		if len(objs) >= 3 {
			groups = append(groups, ObjectGroup{
				Objects:      objs,
				Relationship: "repeated",
			})
		}
	}

	return groups
}

func areSymmetric(obj1, obj2 Object) bool {
	// Check if objects are mirror images
	if obj1.Size != obj2.Size || obj1.Color != obj2.Color {
		return false
	}

	// Check if centroids are symmetric around grid center
	// This is a simplified check
	return obj1.Symmetry != NoSymmetry && obj2.Symmetry != NoSymmetry
}

func analyzeScene(grid [][]int, objects []Object, groups []ObjectGroup) *SceneDescription {
	scene := &SceneDescription{}

	// Detect layout
	scene.Layout = detectLayout(objects, len(grid), len(grid[0]))

	// Detect symmetries
	scene.Symmetries = detectSceneSymmetries(grid)

	// Detect patterns
	scene.Patterns = detectScenePatterns(groups)

	// Calculate complexity
	scene.Complexity = calculateComplexity(objects, groups)

	// Count layers
	scene.NumLayers = countNestingLayers(objects)

	return scene
}

func detectLayout(objects []Object, h, w int) string {
	if len(objects) == 0 {
		return "empty"
	}

	// Check if objects form a grid
	if formsGrid(objects) {
		return "grid"
	}

	// Check if centered
	centerR, centerC := h/2, w/2
	allCentered := true
	for _, obj := range objects {
		dr := obj.Centroid.Row - centerR
		dc := obj.Centroid.Col - centerC
		if dr*dr+dc*dc > (h*h+w*w)/16 {
			allCentered = false
			break
		}
	}
	if allCentered {
		return "centered"
	}

	// Check if on border
	onBorder := 0
	for _, obj := range objects {
		if obj.Bounds.MinRow == 0 || obj.Bounds.MaxRow == h-1 ||
			obj.Bounds.MinCol == 0 || obj.Bounds.MaxCol == w-1 {
			onBorder++
		}
	}
	if onBorder > len(objects)/2 {
		return "border"
	}

	return "scattered"
}

func formsGrid(objects []Object) bool {
	if len(objects) < 4 {
		return false
	}

	// Check if objects are regularly spaced
	rows := make(map[int]int)
	cols := make(map[int]int)
	for _, obj := range objects {
		rows[obj.Centroid.Row]++
		cols[obj.Centroid.Col]++
	}

	return len(rows) >= 2 && len(cols) >= 2
}

func detectSceneSymmetries(grid [][]int) []string {
	symmetries := []string{}

	if isHorizontallySymmetric(grid) {
		symmetries = append(symmetries, "horizontal")
	}
	if isVerticallySymmetric(grid) {
		symmetries = append(symmetries, "vertical")
	}
	if isRotationallySymmetric(grid) {
		symmetries = append(symmetries, "rotational")
	}

	return symmetries
}

func isHorizontallySymmetric(grid [][]int) bool {
	h := len(grid)
	if h == 0 {
		return true
	}
	w := len(grid[0])

	for i := 0; i < h; i++ {
		for j := 0; j < w/2; j++ {
			if grid[i][j] != grid[i][w-1-j] {
				return false
			}
		}
	}
	return true
}

func isVerticallySymmetric(grid [][]int) bool {
	h := len(grid)
	if h == 0 {
		return true
	}
	w := len(grid[0])

	for i := 0; i < h/2; i++ {
		for j := 0; j < w; j++ {
			if grid[i][j] != grid[h-1-i][j] {
				return false
			}
		}
	}
	return true
}

func isRotationallySymmetric(grid [][]int) bool {
	rotated := rot180(grid)
	return equalGrid(grid, rotated)
}

func detectScenePatterns(groups []ObjectGroup) []string {
	patterns := []string{}

	for _, group := range groups {
		if group.Relationship == "repeated" && len(group.Objects) >= 3 {
			patterns = append(patterns, "repeating")
		}
		if group.Relationship == "symmetric" {
			patterns = append(patterns, "symmetric")
		}
		if group.Relationship == "aligned_horizontal" || group.Relationship == "aligned_vertical" {
			patterns = append(patterns, "aligned")
		}
	}

	return patterns
}

func calculateComplexity(objects []Object, groups []ObjectGroup) int {
	complexity := 0

	// Base complexity from object count
	complexity += len(objects)

	// Add complexity for groups
	complexity += len(groups) * 2

	// Add complexity for varied colors
	colors := make(map[int]bool)
	for _, obj := range objects {
		colors[obj.Color] = true
	}
	complexity += len(colors)

	// Normalize to 0-10 scale
	if complexity > 10 {
		complexity = 10
	}

	return complexity
}

func countNestingLayers(objects []Object) int {
	// Count maximum nesting depth
	maxDepth := 1

	for i := 0; i < len(objects); i++ {
		for j := 0; j < len(objects); j++ {
			if i != j && isNested(objects[i], objects[j]) {
				maxDepth = 2
				// Could extend to deeper nesting
			}
		}
	}

	return maxDepth
}

func isNested(inner, outer Object) bool {
	return inner.Bounds.MinRow >= outer.Bounds.MinRow &&
		inner.Bounds.MaxRow <= outer.Bounds.MaxRow &&
		inner.Bounds.MinCol >= outer.Bounds.MinCol &&
		inner.Bounds.MaxCol <= outer.Bounds.MaxCol &&
		inner.Size < outer.Size
}

// HierarchicalTransformOp applies transformations at different abstraction levels
type HierarchicalTransformOp struct {
	Level     AbstractionLevel
	Transform func(*HierarchicalRepresentation) *HierarchicalRepresentation
}

func (op *HierarchicalTransformOp) Apply(grid [][]int) [][]int {
	hr := BuildHierarchy(grid, 0)
	transformed := op.Transform(hr)
	return transformed.Pixels
}

func (op *HierarchicalTransformOp) Name() string {
	return "hierarchical_transform"
}
