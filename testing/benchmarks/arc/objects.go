package arc

import (
	"sort"
)

// Coord represents a grid position
type Coord struct {
	Row, Col int
}

// Object represents a connected component in the grid with properties
type Object struct {
	Pixels   []Coord
	Color    int
	Bounds   Rectangle
	Size     int
	Centroid Coord
	IsConvex bool
	Symmetry SymmetryType
	Holes    int
}

type Rectangle struct {
	MinRow, MinCol, MaxRow, MaxCol int
}

type SymmetryType int

const (
	NoSymmetry SymmetryType = iota
	HorizontalSymmetry
	VerticalSymmetry
	BothSymmetry
	RotationalSymmetry
)

// DetectObjects finds all connected components in the grid
func DetectObjects(grid [][]int, ignoreColor int) []Object {
	if len(grid) == 0 {
		return nil
	}
	h, w := len(grid), len(grid[0])
	visited := make([][]bool, h)
	for i := range visited {
		visited[i] = make([]bool, w)
	}

	var objects []Object
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if !visited[i][j] && grid[i][j] != ignoreColor {
				obj := floodFill(grid, visited, i, j, grid[i][j])
				objects = append(objects, obj)
			}
		}
	}
	return objects
}

func floodFill(grid [][]int, visited [][]bool, startR, startC, color int) Object {
	h, w := len(grid), len(grid[0])
	pixels := []Coord{}
	stack := []Coord{{startR, startC}}
	visited[startR][startC] = true

	minR, minC := startR, startC
	maxR, maxC := startR, startC

	for len(stack) > 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		pixels = append(pixels, curr)

		if curr.Row < minR {
			minR = curr.Row
		}
		if curr.Row > maxR {
			maxR = curr.Row
		}
		if curr.Col < minC {
			minC = curr.Col
		}
		if curr.Col > maxC {
			maxC = curr.Col
		}

		// 4-connectivity
		dirs := []Coord{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
		for _, d := range dirs {
			nr, nc := curr.Row+d.Row, curr.Col+d.Col
			if nr >= 0 && nr < h && nc >= 0 && nc < w &&
				!visited[nr][nc] && grid[nr][nc] == color {
				visited[nr][nc] = true
				stack = append(stack, Coord{nr, nc})
			}
		}
	}

	// Calculate centroid
	sumR, sumC := 0, 0
	for _, p := range pixels {
		sumR += p.Row
		sumC += p.Col
	}
	centroid := Coord{sumR / len(pixels), sumC / len(pixels)}

	return Object{
		Pixels:   pixels,
		Color:    color,
		Bounds:   Rectangle{minR, minC, maxR, maxC},
		Size:     len(pixels),
		Centroid: centroid,
		Symmetry: detectSymmetry(pixels, minR, minC, maxR, maxC),
	}
}

func detectSymmetry(pixels []Coord, minR, minC, maxR, maxC int) SymmetryType {
	pixelSet := make(map[Coord]bool)
	for _, p := range pixels {
		pixelSet[p] = true
	}

	centerR := (minR + maxR) / 2
	centerC := (minC + maxC) / 2

	hasH := true
	hasV := true

	for _, p := range pixels {
		// Check horizontal symmetry (flip across vertical axis)
		mirrorH := Coord{p.Row, 2*centerC - p.Col}
		if !pixelSet[mirrorH] {
			hasH = false
		}

		// Check vertical symmetry (flip across horizontal axis)
		mirrorV := Coord{2*centerR - p.Row, p.Col}
		if !pixelSet[mirrorV] {
			hasV = false
		}
	}

	if hasH && hasV {
		return BothSymmetry
	}
	if hasH {
		return HorizontalSymmetry
	}
	if hasV {
		return VerticalSymmetry
	}
	return NoSymmetry
}

// CountObjects returns the number of objects of a specific color
func CountObjects(objects []Object, color int) int {
	count := 0
	for _, obj := range objects {
		if color < 0 || obj.Color == color {
			count++
		}
	}
	return count
}

// FilterObjectsBySize returns objects within a size range
func FilterObjectsBySize(objects []Object, minSize, maxSize int) []Object {
	var filtered []Object
	for _, obj := range objects {
		if obj.Size >= minSize && obj.Size <= maxSize {
			filtered = append(filtered, obj)
		}
	}
	return filtered
}

// SortObjectsBySize sorts objects by size (descending)
func SortObjectsBySize(objects []Object) []Object {
	sorted := make([]Object, len(objects))
	copy(sorted, objects)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Size > sorted[j].Size
	})
	return sorted
}

// GetObjectColors returns unique colors present in objects
func GetObjectColors(objects []Object) []int {
	colorSet := make(map[int]bool)
	for _, obj := range objects {
		colorSet[obj.Color] = true
	}
	colors := make([]int, 0, len(colorSet))
	for c := range colorSet {
		colors = append(colors, c)
	}
	sort.Ints(colors)
	return colors
}

// SpatialRelation describes the relationship between two objects
type SpatialRelation int

const (
	Above SpatialRelation = iota
	Below
	LeftOf
	RightOf
	Contains
	Inside
	Overlaps
	Adjacent
	Disjoint
)

// GetSpatialRelation determines the spatial relationship between two objects
func GetSpatialRelation(a, b Object) SpatialRelation {
	// Check containment
	if a.Bounds.MinRow <= b.Bounds.MinRow && a.Bounds.MaxRow >= b.Bounds.MaxRow &&
		a.Bounds.MinCol <= b.Bounds.MinCol && a.Bounds.MaxCol >= b.Bounds.MaxCol {
		return Contains
	}
	if b.Bounds.MinRow <= a.Bounds.MinRow && b.Bounds.MaxRow >= a.Bounds.MaxRow &&
		b.Bounds.MinCol <= a.Bounds.MinCol && b.Bounds.MaxCol >= a.Bounds.MaxCol {
		return Inside
	}

	// Check overlap
	if !(a.Bounds.MaxRow < b.Bounds.MinRow || a.Bounds.MinRow > b.Bounds.MaxRow ||
		a.Bounds.MaxCol < b.Bounds.MinCol || a.Bounds.MinCol > b.Bounds.MaxCol) {
		return Overlaps
	}

	// Check adjacency (within 1 cell)
	rowGap := 0
	if a.Bounds.MaxRow < b.Bounds.MinRow {
		rowGap = b.Bounds.MinRow - a.Bounds.MaxRow
	} else if b.Bounds.MaxRow < a.Bounds.MinRow {
		rowGap = a.Bounds.MinRow - b.Bounds.MaxRow
	}
	colGap := 0
	if a.Bounds.MaxCol < b.Bounds.MinCol {
		colGap = b.Bounds.MinCol - a.Bounds.MaxCol
	} else if b.Bounds.MaxCol < a.Bounds.MinCol {
		colGap = a.Bounds.MinCol - b.Bounds.MaxCol
	}
	if rowGap <= 1 && colGap <= 1 {
		return Adjacent
	}

	// Directional relations
	if a.Centroid.Row < b.Centroid.Row {
		return Above
	}
	if a.Centroid.Row > b.Centroid.Row {
		return Below
	}
	if a.Centroid.Col < b.Centroid.Col {
		return LeftOf
	}
	if a.Centroid.Col > b.Centroid.Col {
		return RightOf
	}

	return Disjoint
}
