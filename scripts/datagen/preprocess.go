package datagen

import (
	"image"
	"image/color"
	_ "image/png"
	"io/fs"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"
)

const (
	GridSize = 20 // Downsample images to 20x20 grids
)

// ProcessImage converts an image file to a downsampled integer grid.
func ProcessImage(path string) ([][]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	src, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	// Create a new 20x20 image
	dst := image.NewRGBA(image.Rect(0, 0, GridSize, GridSize))

	// Resize the source image to the destination image
	draw.NearestNeighbor.Scale(dst, dst.Rect, src, src.Bounds(), draw.Over, nil)

	// Convert the resized image to an integer grid
	grid := make([][]int, GridSize)
	for y := 0; y < GridSize; y++ {
		grid[y] = make([]int, GridSize)
		for x := 0; x < GridSize; x++ {
			grid[y][x] = quantizeColor(dst.At(x, y))
		}
	}

	return grid, nil
}

// quantizeColor converts a color to a simple integer (0-3).
func quantizeColor(c color.Color) int {
	gray := color.GrayModel.Convert(c).(color.Gray)
	switch {
	case gray.Y < 64:
		return 3 // Black
	case gray.Y < 128:
		return 2 // Dark Gray
	case gray.Y < 192:
		return 1 // Light Gray
	default:
		return 0 // White
	}
}

// LoadCharacterGrids processes all images in a directory.
func LoadCharacterGrids(dir string) (map[string][][][]int, error) {
	charMap := make(map[string][][][]int)

	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || filepath.Ext(path) != ".png" {
			return nil
		}

		// Character name is the parent directory's name
		charName := filepath.Base(filepath.Dir(path))
		grid, err := ProcessImage(path)
		if err != nil {
			log.Printf("Failed to process %s: %v", path, err)
			return nil
		}

		charMap[charName] = append(charMap[charName], grid)
		return nil
	})

	if err != nil {
		return nil, err
	}

	return charMap, nil
}
