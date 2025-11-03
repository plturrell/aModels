//go:build gemma_integration

package ai

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMultiHeadAttentionRealWeights(t *testing.T) {
	modelDir := os.Getenv("GEMMA_MODEL_PATH")
	if strings.TrimSpace(modelDir) == "" {
		modelDir = filepath.Clean("../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1")
	}

	info, err := os.Stat(modelDir)
	if err != nil || !info.IsDir() {
		t.Skipf("model directory %q not available: %v", modelDir, err)
	}

	vg, err := LoadVaultGemmaFromSafetensors(modelDir)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	inputIDs := []int{2, 7, 15, 3}
	hidden := vg.embedTokens(inputIDs)
	layer := &vg.Layers[0]
	t.Logf("num_heads=%d head_dim=%d hidden=%d", vg.Config.NumHeads, vg.Config.HeadDim, vg.Config.HiddenSize)
	t.Logf("WQ rows=%d cols=%d stride=%d", layer.SelfAttention.WQ.Rows, layer.SelfAttention.WQ.Cols, layer.SelfAttention.WQ.Stride)
	normed := vg.rmsNorm(hidden, layer.LayerNorm1)
	attnOut := vg.multiHeadAttention(normed, layer.SelfAttention)

	golden := []float64{
		0.030054343190388938,
		-0.2630642396299889,
		0.0826062337221004,
		-0.0432988495469831,
		0.06289537374368735,
		0.08902498084220628,
		-0.17203374894201587,
		-0.19543297317387473,
	}

	for i, want := range golden {
		if i >= len(attnOut.Data) {
			t.Fatalf("output shorter than expected: have %d need %d", len(attnOut.Data), len(golden))
		}
		if math.Abs(attnOut.Data[i]-want) > 1e-9 {
			t.Fatalf("mismatch at %d: got %.12f want %.12f", i, attnOut.Data[i], want)
		}
	}
}

func TestForwardLogitsGolden(t *testing.T) {
	modelDir := os.Getenv("GEMMA_MODEL_PATH")
	if strings.TrimSpace(modelDir) == "" {
		modelDir = filepath.Clean("../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1")
	}

	info, err := os.Stat(modelDir)
	if err != nil || !info.IsDir() {
		t.Skipf("model directory %q not available: %v", modelDir, err)
	}

	vg, err := LoadVaultGemmaFromSafetensors(modelDir)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	inputIDs := []int{vg.Config.BOSTokenID, 42, 17, 3}
	layer := &vg.Layers[0]
	t.Logf("num_heads=%d head_dim=%d hidden=%d", vg.Config.NumHeads, vg.Config.HeadDim, vg.Config.HiddenSize)
	t.Logf("WQ rows=%d cols=%d stride=%d", layer.SelfAttention.WQ.Rows, layer.SelfAttention.WQ.Cols, layer.SelfAttention.WQ.Stride)
	logits, err := vg.Forward(inputIDs)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	lastRow := logits.Data[(logits.Rows-1)*logits.Stride : logits.Rows*logits.Stride]

	maxVal := lastRow[0]
	maxIdx := 0
	for i, v := range lastRow {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	t.Logf("max logit %.6f at index %d", maxVal, maxIdx)

	const expectedMaxIdx = 10328
	const expectedMaxVal = 11.192468630034164

	if maxIdx != expectedMaxIdx {
		t.Fatalf("max index mismatch: got %d want %d", maxIdx, expectedMaxIdx)
	}
	if math.Abs(maxVal-expectedMaxVal) > 1e-6 {
		t.Fatalf("max value mismatch: got %.9f want %.9f", maxVal, expectedMaxVal)
	}
}
