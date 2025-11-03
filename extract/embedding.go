package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
)

func generateEmbedding(ctx context.Context, sql string) ([]float32, error) {
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sql.py", "--sql", sql)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
	}

	return embedding, nil
}
