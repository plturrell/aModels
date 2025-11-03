package ai

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

const (
	tensorReadChunk = 1 << 22 // 4,194,304 elements (~16 MB for float32)
	float32Size     = 4
	float16Size     = 2
)

// SafetensorsHeader represents the metadata header in a safetensors file
type SafetensorsHeader struct {
	Metadata map[string]TensorInfo `json:"__metadata__"`
	Tensors  map[string]TensorInfo
}

type TensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// LoadVaultGemmaFromSafetensors loads a VaultGemma model from safetensors format
func LoadVaultGemmaFromSafetensors(modelPath string) (*VaultGemma, error) {
	// Load config
	configPath := filepath.Join(modelPath, "config.json")
	vg, err := NewVaultGemma(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	// Load weights from safetensors
	indexPath := filepath.Join(modelPath, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}

		data, err := os.ReadFile(indexPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read weight index: %v", err)
		}

		if err := json.Unmarshal(data, &index); err != nil {
			return nil, fmt.Errorf("failed to parse weight index: %v", err)
		}

		if len(index.WeightMap) == 0 {
			return nil, fmt.Errorf("weight index is empty: %s", indexPath)
		}

		shards := make([]string, 0, len(index.WeightMap))
		seen := make(map[string]struct{}, len(index.WeightMap))
		for _, shard := range index.WeightMap {
			if _, exists := seen[shard]; exists {
				continue
			}
			seen[shard] = struct{}{}
			shards = append(shards, shard)
		}

		sort.Strings(shards)

		for _, shard := range shards {
			shardPath := filepath.Join(modelPath, shard)
			if err := vg.loadWeights(shardPath); err != nil {
				return nil, fmt.Errorf("failed to load weights from %s: %v", shard, err)
			}
		}
	} else {
		weightsPath := filepath.Join(modelPath, "model.safetensors")
		if err := vg.loadWeights(weightsPath); err != nil {
			return nil, fmt.Errorf("failed to load weights: %v", err)
		}
	}

	vg.tieOutputWeightsIfNeeded()

	return vg, nil
}

func (vg *VaultGemma) loadWeights(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open weights file: %v", err)
	}
	defer f.Close()

	// Read header size (first 8 bytes, little-endian)
	var headerSize int64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return fmt.Errorf("failed to read header size: %v", err)
	}

	// Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return fmt.Errorf("failed to read header: %v", err)
	}

	// Parse header
	var header map[string]TensorInfo
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return fmt.Errorf("failed to parse header: %v", err)
	}

	// Current position in file (after header)
	dataStart := int64(8 + headerSize)

	// Load each tensor
	for name, info := range header {
		if name == "__metadata__" {
			continue
		}

		// Read tensor data
		tensorData, err := vg.readTensor(f, info, dataStart)
		if err != nil {
			return fmt.Errorf("failed to read tensor %s: %v", name, err)
		}

		// Assign to model
		if err := vg.assignTensor(name, tensorData, info.Shape); err != nil {
			return fmt.Errorf("failed to assign tensor %s: %v", name, err)
		}
	}

	return nil
}

func (vg *VaultGemma) readTensor(f *os.File, info TensorInfo, dataStart int64) ([]float64, error) {
	// Seek to tensor data
	offset := dataStart + info.DataOffsets[0]
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}

	// Calculate number of elements
	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	// Read based on dtype
	switch info.DType {
	case "F32", "float32":
		return vg.readFloat32Tensor(f, numElements)
	case "F16", "float16":
		return vg.readFloat16Tensor(f, numElements)
	case "BF16", "bfloat16":
		return vg.readBFloat16Tensor(f, numElements)
	default:
		return nil, fmt.Errorf("unsupported dtype: %s", info.DType)
	}
}

func (vg *VaultGemma) readFloat32Tensor(f *os.File, numElements int) ([]float64, error) {
	result := make([]float64, numElements)
	buffer := make([]byte, tensorReadChunk*float32Size)

	processed := 0
	for processed < numElements {
		chunkSize := min(tensorReadChunk, numElements-processed)
		bytesNeeded := chunkSize * float32Size

		if _, err := io.ReadFull(f, buffer[:bytesNeeded]); err != nil {
			return nil, err
		}

		convertFloat32Chunk(result[processed:processed+chunkSize], buffer[:bytesNeeded])
		processed += chunkSize
	}

	return result, nil
}

func (vg *VaultGemma) readFloat16Tensor(f *os.File, numElements int) ([]float64, error) {
	result := make([]float64, numElements)
	buffer := make([]byte, tensorReadChunk*float16Size)

	processed := 0
	for processed < numElements {
		chunkSize := min(tensorReadChunk, numElements-processed)
		bytesNeeded := chunkSize * float16Size

		if _, err := io.ReadFull(f, buffer[:bytesNeeded]); err != nil {
			return nil, err
		}

		convertFloat16Chunk(result[processed:processed+chunkSize], buffer[:bytesNeeded])
		processed += chunkSize
	}

	return result, nil
}

func (vg *VaultGemma) readBFloat16Tensor(f *os.File, numElements int) ([]float64, error) {
	result := make([]float64, numElements)
	buffer := make([]byte, tensorReadChunk*float16Size)

	processed := 0
	for processed < numElements {
		chunkSize := min(tensorReadChunk, numElements-processed)
		bytesNeeded := chunkSize * float16Size

		if _, err := io.ReadFull(f, buffer[:bytesNeeded]); err != nil {
			return nil, err
		}

		convertBFloat16Chunk(result[processed:processed+chunkSize], buffer[:bytesNeeded])
		processed += chunkSize
	}

	return result, nil
}

// Convert float16 to float64
func float16ToFloat64(bits uint16) float64 {
	sign := uint32((bits >> 15) & 1)
	exponent := uint32((bits >> 10) & 0x1F)
	mantissa := uint32(bits & 0x3FF)

	if exponent == 0 {
		if mantissa == 0 {
			return 0.0
		}
		// Subnormal
		return float64(float32(mantissa) / 1024.0 / 16384.0)
	}

	if exponent == 31 {
		// Infinity or NaN
		return 0.0 // Simplified handling
	}

	// Normalized
	exponent = exponent - 15 + 127
	bits32 := (sign << 31) | (exponent << 23) | (mantissa << 13)
	return float64(float32FromBits(bits32))
}

// Convert bfloat16 to float64
func bfloat16ToFloat64(bits uint16) float64 {
	// BF16 is just the top 16 bits of float32
	bits32 := uint32(bits) << 16
	return float64(float32FromBits(bits32))
}

func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func convertFloat32Chunk(dst []float64, src []byte) {
	for i := range dst {
		bits := binary.LittleEndian.Uint32(src[i*float32Size : (i+1)*float32Size])
		dst[i] = float64(math.Float32frombits(bits))
	}
}

func convertFloat16Chunk(dst []float64, src []byte) {
	for i := range dst {
		bits := binary.LittleEndian.Uint16(src[i*float16Size : (i+1)*float16Size])
		dst[i] = float16ToFloat64(bits)
	}
}

func convertBFloat16Chunk(dst []float64, src []byte) {
	for i := range dst {
		bits := binary.LittleEndian.Uint16(src[i*float16Size : (i+1)*float16Size])
		dst[i] = bfloat16ToFloat64(bits)
	}
}

func (vg *VaultGemma) assignTensor(name string, data []float64, shape []int) error {
	// Map tensor names to model components
	// VaultGemma naming convention: model.layers.{i}.{component}.{weight/bias}

	switch {
	case name == "model.embed_tokens.weight":
		return vg.assignEmbedding(data, shape)

	case contains(name, "self_attn.q_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].SelfAttention.WQ, data, shape)

	case contains(name, "self_attn.k_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].SelfAttention.WK, data, shape)

	case contains(name, "self_attn.v_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].SelfAttention.WV, data, shape)

	case contains(name, "self_attn.o_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].SelfAttention.WO, data, shape)

	case contains(name, "mlp.gate_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].FeedForward.W3, data, shape)

	case contains(name, "mlp.up_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].FeedForward.W1, data, shape)

	case contains(name, "mlp.down_proj.weight"):
		layer := extractLayerNum(name)
		return vg.assignWeight(&vg.Layers[layer].FeedForward.W2, data, shape)

	case contains(name, "input_layernorm.weight"):
		layer := extractLayerNum(name)
		vg.Layers[layer].LayerNorm1.Weight = data
		return nil

	case contains(name, "post_attention_layernorm.weight"):
		layer := extractLayerNum(name)
		vg.Layers[layer].LayerNorm2.Weight = data
		return nil

	case name == "lm_head.weight":
		return vg.assignWeight(&vg.Output.Weights, data, shape)

	default:
		// Skip unknown tensors
		return nil
	}
}

func (vg *VaultGemma) assignEmbedding(data []float64, shape []int) error {
	if len(shape) != 2 {
		return fmt.Errorf("invalid embedding shape: %v", shape)
	}
	rows, cols := shape[0], shape[1]
	if vg.Embed == nil {
		vg.Embed = &EmbeddingLayer{}
	}
	if vg.Embed.Weights == nil || vg.Embed.Weights.Rows != rows || vg.Embed.Weights.Cols != cols {
		vg.Embed.Weights = util.NewMatrix64(rows, cols)
	}
	weights := vg.Embed.Weights
	if len(weights.Data) != len(data) {
		weights.Data = make([]float64, len(data))
	}
	weights.Rows = rows
	weights.Cols = cols
	weights.Stride = cols
	copy(weights.Data, data)
	return nil
}

func (vg *VaultGemma) assignWeight(matrix **util.Matrix64, data []float64, shape []int) error {
	if len(shape) != 2 {
		return fmt.Errorf("invalid weight shape: %v", shape)
	}
	rows, cols := shape[0], shape[1]

	if *matrix == nil {
		*matrix = util.NewMatrix64(rows, cols)
	}
	dest := *matrix

	expectedRows := dest.Rows
	expectedCols := dest.Cols

	switch {
	case expectedRows == rows && expectedCols == cols:
		if len(dest.Data) != len(data) {
			dest.Data = make([]float64, len(data))
		}
		copy(dest.Data, data)
		dest.Stride = expectedCols
	case expectedRows == cols && expectedCols == rows:
		if len(dest.Data) != len(data) {
			dest.Data = make([]float64, len(data))
		}
		dest.Stride = expectedCols
		for r := 0; r < expectedRows; r++ {
			for c := 0; c < expectedCols; c++ {
				dest.Data[r*dest.Stride+c] = data[c*cols+r]
			}
		}
	default:
		dest.Rows = rows
		dest.Cols = cols
		dest.Stride = cols
		dest.Data = make([]float64, len(data))
		copy(dest.Data, data)
	}

	return nil
}

func (vg *VaultGemma) tieOutputWeightsIfNeeded() {
	embed := vg.Embed
	if embed == nil || embed.Weights == nil {
		return
	}
	if vg.Output == nil {
		vg.Output = &OutputLayer{Weights: util.NewMatrix64(embed.Weights.Cols, embed.Weights.Rows)}
	}
	weights := vg.Output.Weights
	if weights == nil || len(weights.Data) == 0 || allZero(weights.Data) {
		rows := embed.Weights.Cols
		cols := embed.Weights.Rows
		if weights == nil || weights.Rows != rows || weights.Cols != cols {
			vg.Output.Weights = util.NewMatrix64(rows, cols)
			weights = vg.Output.Weights
		}
		if len(weights.Data) != rows*cols {
			weights.Data = make([]float64, rows*cols)
		}
		weights.Rows = rows
		weights.Cols = cols
		weights.Stride = cols
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				weights.Data[r*weights.Stride+c] = embed.Weights.Data[c*embed.Weights.Stride+r]
			}
		}
	}
}

func allZero(data []float64) bool {
	for _, v := range data {
		if v != 0 {
			return false
		}
	}
	return true
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[len(s)-len(substr):] == substr ||
		len(s) > len(substr) && s[:len(substr)] == substr ||
		findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func extractLayerNum(name string) int {
	// Extract layer number from name like "model.layers.0.self_attn..."
	var layer int
	fmt.Sscanf(name, "model.layers.%d", &layer)
	return layer
}
