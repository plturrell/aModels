package main

import (
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
)

func main() {
	modelDir := os.Getenv("GEMMA_MODEL_PATH")
	if strings.TrimSpace(modelDir) == "" {
		modelDir = filepath.Clean("../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1")
	}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	logger.Info("vaultgemma-debug", "event", "load", "model_path", modelDir)

	model, err := ai.LoadVaultGemmaFromSafetensors(modelDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}

	inputIDs := []int{2, 7, 15, 3}
	embedded, normed, attnOut, err := model.DebugAttentionPass(inputIDs, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "debug attention: %v\n", err)
		os.Exit(1)
	}

	firstAttn := make([]float64, 0, 8)
	for i := 0; i < 8 && i < len(attnOut.Data); i++ {
		firstAttn = append(firstAttn, attnOut.Data[i])
	}

	attnNorm := l2Norm(attnOut.Data)
	embedNorm := 0.0
	if len(embedded.Data) > 0 {
		embedNorm = l2Norm(embedded.Data[:embedded.Cols])
	}
	rmsNorm := 0.0
	if len(normed.Data) > 0 {
		rmsNorm = l2Norm(normed.Data[:normed.Cols])
	}

	logger.Info("vaultgemma-debug", "event", "attention_sample",
		"first_attention", firstAttn,
		"attention_norm", attnNorm,
		"embedding_norm", embedNorm,
		"rms_norm", rmsNorm,
		"input_ids", inputIDs,
		"layer_index", 0,
	)
}

func l2Norm(vec []float64) float64 {
	sum := 0.0
	for _, v := range vec {
		sum += v * v
	}
	return math.Sqrt(sum)
}
