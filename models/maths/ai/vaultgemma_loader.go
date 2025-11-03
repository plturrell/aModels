package ai

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// SafetensorsHeader represents the metadata header in a safetensors file
type SafetensorsHeader struct {
	Metadata map[string]TensorInfo `json:"__metadata__"`
	Tensors  map[string]TensorInfo
}

type TensorInfo struct {
	DType      string   `json:"dtype"`
	Shape      []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// LoadVaultGemmaFromSafetensors loads a VaultGemma model from safetensors format
func LoadVaultGemmaFromSafetensors(modelPath string) (*VaultGemma, error) {
	// Load config
	configPath := modelPath + "/config.json"
	vg, err := NewVaultGemma(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	// Load weights from safetensors
	weightsPath := modelPath + "/model.safetensors"
	if err := vg.loadWeights(weightsPath); err != nil {
		return nil, fmt.Errorf("failed to load weights: %v", err)
	}

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
	data := make([]float64, numElements)
	for i := 0; i < numElements; i++ {
		var val float32
		if err := binary.Read(f, binary.LittleEndian, &val); err != nil {
			return nil, err
		}
		data[i] = float64(val)
	}
	return data, nil
}

func (vg *VaultGemma) readFloat16Tensor(f *os.File, numElements int) ([]float64, error) {
	data := make([]float64, numElements)
	for i := 0; i < numElements; i++ {
		var val uint16
		if err := binary.Read(f, binary.LittleEndian, &val); err != nil {
			return nil, err
		}
		data[i] = float16ToFloat64(val)
	}
	return data, nil
}

func (vg *VaultGemma) readBFloat16Tensor(f *os.File, numElements int) ([]float64, error) {
	data := make([]float64, numElements)
	for i := 0; i < numElements; i++ {
		var val uint16
		if err := binary.Read(f, binary.LittleEndian, &val); err != nil {
			return nil, err
		}
		data[i] = bfloat16ToFloat64(val)
	}
	return data, nil
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
	// Convert bits to float32 using math package
	return float32(bits) // Simplified for now
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
	vg.Embed.Weights = &util.Matrix64{
		Data: data,
		Rows: shape[0],
		Cols: shape[1],
	}
	return nil
}

func (vg *VaultGemma) assignWeight(matrix **util.Matrix64, data []float64, shape []int) error {
	if len(shape) != 2 {
		return fmt.Errorf("invalid weight shape: %v", shape)
	}
	*matrix = &util.Matrix64{
		Data: data,
		Rows: shape[0],
		Cols: shape[1],
	}
	return nil
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
