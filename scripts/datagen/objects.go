package datagen

// Coord represents a grid position
type Coord struct {
	Row, Col int
}

// Object represents a connected component in the grid
type Object struct {
	Pixels []Coord
	Color  int
	Size   int
}

// DetectObjects finds all connected components in the grid.
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

	for len(stack) > 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		pixels = append(pixels, curr)

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

	return Object{
		Pixels: pixels,
		Color:  color,
		Size:   len(pixels),
	}
}
