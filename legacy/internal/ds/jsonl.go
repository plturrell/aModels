package ds

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
)

// ReadJSONL reads a JSON Lines file and decodes each line into v, which must be a pointer to a zero value
// of the element type. It returns the number of decoded records.
func ReadJSONL[T any](path string, limit int, fn func(*T) error) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	dec := json.NewDecoder(r)
	count := 0
	for {
		if limit > 0 && count >= limit {
			break
		}
		var v T
		if err := dec.Decode(&v); err != nil {
			if err == io.EOF {
				break
			}
			return count, err
		}
		if err := fn(&v); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}
