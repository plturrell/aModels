package ai

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// VaultGemma - 1B parameter differential privacy language model
// Pure Go implementation using native tensor operations
type VaultGemma struct {
	Config VaultGemmaConfig
	Layers []TransformerLayer
	Embed  *EmbeddingLayer
	Output *OutputLayer
}

type VaultGemmaConfig struct {
	HiddenSize       int     `json:"hidden_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	VocabSize        int     `json:"vocab_size"`
	MaxPositionEmbs  int     `json:"max_position_embeddings"`
	IntermediateSize int     `json:"intermediate_size"`
	HeadDim          int     `json:"head_dim"`
	RMSNormEps       float64 `json:"rms_norm_eps"`
	RopeTheta        float64 `json:"rope_theta"`
	EOSTokenID       int     `json:"eos_token_id"`
	BOSTokenID       int     `json:"bos_token_id"`
}

type TransformerLayer struct {
	SelfAttention *MultiHeadAttention
	FeedForward   *FeedForwardNetwork
	LayerNorm1    *RMSNorm
	LayerNorm2    *RMSNorm
}

type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	WQ       *util.Matrix64 // Query projection
	WK       *util.Matrix64 // Key projection
	WV       *util.Matrix64 // Value projection
	WO       *util.Matrix64 // Output projection
	Cache    AttentionCache // cached keys/values for autoregressive decoding
}

// AttentionCache stores cached key/value projections for autoregressive decoding.
type AttentionCache struct {
	Keys   *util.Matrix64
	Values *util.Matrix64
}

type FeedForwardNetwork struct {
	W1 *util.Matrix64 // Up projection
	W2 *util.Matrix64 // Down projection
	W3 *util.Matrix64 // Gate projection
}

type RMSNorm struct {
	Weight []float64
	Eps    float64
}

type EmbeddingLayer struct {
	Weights *util.Matrix64 // [VocabSize x HiddenSize]
}

type OutputLayer struct {
	Weights *util.Matrix64 // [HiddenSize x VocabSize]
}

// NewVaultGemma creates a new VaultGemma model
func NewVaultGemma(configPath string) (*VaultGemma, error) {
	// Load config
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %v", err)
	}

	var config VaultGemmaConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %v", err)
	}

	if config.RopeTheta == 0 {
		config.RopeTheta = 10000.0
	}
	if config.EOSTokenID == 0 {
		config.EOSTokenID = 1
	}
	if config.BOSTokenID == 0 {
		config.BOSTokenID = 2
	}
	if config.HeadDim == 0 && config.NumHeads > 0 {
		config.HeadDim = config.HiddenSize / config.NumHeads
	}

	vg := &VaultGemma{
		Config: config,
		Layers: make([]TransformerLayer, config.NumLayers),
	}

	// Initialize layers
	vg.initializeLayers()

	return vg, nil
}

func (vg *VaultGemma) initializeLayers() {
	hiddenSize := vg.Config.HiddenSize
	numHeads := vg.Config.NumHeads
	headDim := vg.Config.HeadDim
	intermediateSize := vg.Config.IntermediateSize

	// Initialize embedding
	vg.Embed = &EmbeddingLayer{
		Weights: util.NewMatrix64(vg.Config.VocabSize, hiddenSize),
	}

	// Initialize transformer layers
	for i := 0; i < vg.Config.NumLayers; i++ {
		vg.Layers[i] = TransformerLayer{
			SelfAttention: &MultiHeadAttention{
				NumHeads: numHeads,
				HeadDim:  headDim,
				WQ:       util.NewMatrix64(hiddenSize, numHeads*headDim),
				WK:       util.NewMatrix64(hiddenSize, numHeads*headDim),
				WV:       util.NewMatrix64(hiddenSize, numHeads*headDim),
				WO:       util.NewMatrix64(numHeads*headDim, hiddenSize),
			},
			FeedForward: &FeedForwardNetwork{
				W1: util.NewMatrix64(hiddenSize, intermediateSize),
				W2: util.NewMatrix64(intermediateSize, hiddenSize),
				W3: util.NewMatrix64(hiddenSize, intermediateSize),
			},
			LayerNorm1: &RMSNorm{
				Weight: make([]float64, hiddenSize),
				Eps:    vg.Config.RMSNormEps,
			},
			LayerNorm2: &RMSNorm{
				Weight: make([]float64, hiddenSize),
				Eps:    vg.Config.RMSNormEps,
			},
		}

		// Initialize RMSNorm weights to 1.0
		for j := range vg.Layers[i].LayerNorm1.Weight {
			vg.Layers[i].LayerNorm1.Weight[j] = 1.0
			vg.Layers[i].LayerNorm2.Weight[j] = 1.0
		}
	}

	// Initialize output layer
	vg.Output = &OutputLayer{
		Weights: util.NewMatrix64(hiddenSize, vg.Config.VocabSize),
	}
}

// Forward pass through the model
func (vg *VaultGemma) Forward(inputIDs []int) (*util.Matrix64, error) {
	// 1. Embedding lookup
	hidden := vg.embedTokens(inputIDs)

	// 2. Process through transformer layers
	for i := 0; i < vg.Config.NumLayers; i++ {
		hidden = vg.transformerLayer(hidden, &vg.Layers[i])
	}

	// 3. Output projection
	logits := vg.projectOutput(hidden)

	return logits, nil
}

func (vg *VaultGemma) embedTokens(inputIDs []int) *util.Matrix64 {
	seqLen := len(inputIDs)
	hiddenSize := vg.Config.HiddenSize

	embedded := util.NewMatrix64(seqLen, hiddenSize)

	// Lookup embeddings for each token
	for i, tokenID := range inputIDs {
		if tokenID >= 0 && tokenID < vg.Config.VocabSize {
			// Copy embedding vector
			for j := 0; j < hiddenSize; j++ {
				embedded.Data[i*hiddenSize+j] = vg.Embed.Weights.Data[tokenID*hiddenSize+j]
			}
		}
	}

	return embedded
}

// DebugAttentionPass exposes a pared-down attention path for diagnostics.
// It returns the embedded tokens, the pre-attention RMSNorm output, and the
// resulting multi-head attention output for the requested transformer layer.
func (vg *VaultGemma) DebugAttentionPass(inputIDs []int, layerIndex int) (*util.Matrix64, *util.Matrix64, *util.Matrix64, error) {
	if layerIndex < 0 || layerIndex >= len(vg.Layers) {
		return nil, nil, nil, fmt.Errorf("layer index %d out of range", layerIndex)
	}
	embedded := vg.embedTokens(inputIDs)
	normed := vg.rmsNorm(embedded, vg.Layers[layerIndex].LayerNorm1)
	attn := vg.multiHeadAttention(normed, vg.Layers[layerIndex].SelfAttention)
	return embedded, normed, attn, nil
}

func (vg *VaultGemma) transformerLayer(hidden *util.Matrix64, layer *TransformerLayer) *util.Matrix64 {
	// 1. Pre-attention layer norm
	normed := vg.rmsNorm(hidden, layer.LayerNorm1)

	// 2. Multi-head self-attention
	attnOutput := vg.multiHeadAttention(normed, layer.SelfAttention)

	// 3. Residual connection
	hidden = vg.addResidual(hidden, attnOutput)

	// 4. Pre-FFN layer norm
	normed = vg.rmsNorm(hidden, layer.LayerNorm2)

	// 5. Feed-forward network
	ffnOutput := vg.feedForward(normed, layer.FeedForward)

	// 6. Residual connection
	hidden = vg.addResidual(hidden, ffnOutput)

	return hidden
}

func (vg *VaultGemma) multiHeadAttention(x *util.Matrix64, attn *MultiHeadAttention) *util.Matrix64 {
	return vg.multiHeadAttentionWithCache(x, attn, nil)
}

func (vg *VaultGemma) multiHeadAttentionWithCache(x *util.Matrix64, attn *MultiHeadAttention, cache *AttentionCache) *util.Matrix64 {
	// Project to Q, K, V using matrix multiplication
	// Q = x @ WQ
	QData := util.MatMul(x.Rows, attn.WQ.Cols, x.Cols, x.Data, attn.WQ.Data)
	Q := &util.Matrix64{Data: QData, Rows: x.Rows, Cols: attn.WQ.Cols, Stride: attn.WQ.Cols}

	// K = x @ WK
	KData := util.MatMul(x.Rows, attn.WK.Cols, x.Cols, x.Data, attn.WK.Data)
	K := &util.Matrix64{Data: KData, Rows: x.Rows, Cols: attn.WK.Cols, Stride: attn.WK.Cols}

	// V = x @ WV
	VData := util.MatMul(x.Rows, attn.WV.Cols, x.Cols, x.Data, attn.WV.Data)
	V := &util.Matrix64{Data: VData, Rows: x.Rows, Cols: attn.WV.Cols, Stride: attn.WV.Cols}

	// Apply rotary embeddings to Q and K
	if attn.HeadDim%2 != 0 {
		panic("head dimension must be even for rotary embeddings")
	}
	applyRotaryEmbeddings(Q, attn.NumHeads, attn.HeadDim, vg.Config.RopeTheta)
	applyRotaryEmbeddings(K, attn.NumHeads, attn.HeadDim, vg.Config.RopeTheta)

	keysMat := K
	valuesMat := V
	if cache != nil {
		if cache.Keys != nil {
			keysMat = concatMatrixRows(cache.Keys, K)
		} else {
			keysMat = cloneMatrix(K)
		}
		if cache.Values != nil {
			valuesMat = concatMatrixRows(cache.Values, V)
		} else {
			valuesMat = cloneMatrix(V)
		}
		cache.Keys = cloneMatrix(keysMat)
		cache.Values = cloneMatrix(valuesMat)
	}

	// Scale factor for attention
	scale := 1.0 / math.Sqrt(float64(attn.HeadDim))

	seqLen := x.Rows
	concat := util.NewMatrix64(seqLen, attn.NumHeads*attn.HeadDim)

	for h := 0; h < attn.NumHeads; h++ {
		qHead := sliceHeadMatrix(Q, h, attn.NumHeads, attn.HeadDim)
		kHead := sliceHeadMatrix(keysMat, h, attn.NumHeads, attn.HeadDim)
		vHead := sliceHeadMatrix(valuesMat, h, attn.NumHeads, attn.HeadDim)

		headOutput := tensor.FlashAttention(qHead, kHead, vHead, scale)

		for pos := 0; pos < seqLen; pos++ {
			dst := concat.Data[pos*concat.Stride+h*attn.HeadDim : pos*concat.Stride+(h+1)*attn.HeadDim]
			src := headOutput.Data[pos*headOutput.Stride : pos*headOutput.Stride+attn.HeadDim]
			copy(dst, src)
		}
	}

	// Project back to hidden size: output = attnOutput @ WO
	outputData := util.MatMul(concat.Rows, attn.WO.Cols, concat.Cols, concat.Data, attn.WO.Data)
	output := &util.Matrix64{Data: outputData, Rows: concat.Rows, Cols: attn.WO.Cols, Stride: attn.WO.Cols}

	return output
}

func (vg *VaultGemma) feedForward(x *util.Matrix64, ffn *FeedForwardNetwork) *util.Matrix64 {
	// SwiGLU activation: FFN(x) = (W1(x) * SiLU(W3(x))) @ W2

	// Gate projection: gate = x @ W3
	gateData := util.MatMul(x.Rows, ffn.W3.Cols, x.Cols, x.Data, ffn.W3.Data)
	gate := &util.Matrix64{Data: gateData, Rows: x.Rows, Cols: ffn.W3.Cols}
	gate = vg.silu(gate)

	// Up projection: up = x @ W1
	upData := util.MatMul(x.Rows, ffn.W1.Cols, x.Cols, x.Data, ffn.W1.Data)
	up := &util.Matrix64{Data: upData, Rows: x.Rows, Cols: ffn.W1.Cols}

	// Element-wise multiplication
	hidden := vg.elementwiseMul(up, gate)

	// Down projection: output = hidden @ W2
	outputData := util.MatMul(hidden.Rows, ffn.W2.Cols, hidden.Cols, hidden.Data, ffn.W2.Data)
	output := &util.Matrix64{Data: outputData, Rows: hidden.Rows, Cols: ffn.W2.Cols}

	return output
}

func (vg *VaultGemma) rmsNorm(x *util.Matrix64, norm *RMSNorm) *util.Matrix64 {
	rows := x.Rows
	cols := x.Cols
	output := util.NewMatrix64(rows, cols)

	for i := 0; i < rows; i++ {
		// Compute RMS
		var sumSquares float64
		for j := 0; j < cols; j++ {
			val := x.Data[i*cols+j]
			sumSquares += val * val
		}
		rms := math.Sqrt(sumSquares/float64(cols) + norm.Eps)

		// Normalize and scale
		for j := 0; j < cols; j++ {
			output.Data[i*cols+j] = (x.Data[i*cols+j] / rms) * norm.Weight[j]
		}
	}

	return output
}

func (vg *VaultGemma) silu(x *util.Matrix64) *util.Matrix64 {
	// SiLU(x) = x * sigmoid(x)
	output := util.NewMatrix64(x.Rows, x.Cols)

	for i := 0; i < len(x.Data); i++ {
		val := x.Data[i]
		sigmoid := 1.0 / (1.0 + math.Exp(-val))
		output.Data[i] = val * sigmoid
	}

	return output
}

func (vg *VaultGemma) elementwiseMul(a, b *util.Matrix64) *util.Matrix64 {
	output := util.NewMatrix64(a.Rows, a.Cols)

	for i := 0; i < len(a.Data); i++ {
		output.Data[i] = a.Data[i] * b.Data[i]
	}

	return output
}

func (vg *VaultGemma) addResidual(x, residual *util.Matrix64) *util.Matrix64 {
	output := util.NewMatrix64(x.Rows, x.Cols)

	for i := 0; i < len(x.Data); i++ {
		output.Data[i] = x.Data[i] + residual.Data[i]
	}

	return output
}

func (vg *VaultGemma) projectOutput(hidden *util.Matrix64) *util.Matrix64 {
	// Project to vocabulary size: logits = hidden @ Output.Weights
	logitsData := util.MatMul(hidden.Rows, vg.Output.Weights.Cols, hidden.Cols, hidden.Data, vg.Output.Weights.Data)
	logits := &util.Matrix64{Data: logitsData, Rows: hidden.Rows, Cols: vg.Output.Weights.Cols, Stride: vg.Output.Weights.Cols}
	return logits
}

func applyRotaryEmbeddings(m *util.Matrix64, numHeads, headDim int, theta float64) {
	if m == nil || m.Rows == 0 || headDim%2 != 0 {
		return
	}
	if theta == 0 {
		theta = 10000.0
	}

	half := headDim / 2
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		exponent := float64(2*i) / float64(headDim)
		invFreq[i] = math.Pow(theta, -exponent)
	}

	cosCache := make([]float64, half)
	sinCache := make([]float64, half)

	for pos := 0; pos < m.Rows; pos++ {
		row := m.Data[pos*m.Stride : pos*m.Stride+m.Cols]
		for idx := 0; idx < half; idx++ {
			angle := float64(pos) * invFreq[idx]
			cosCache[idx] = math.Cos(angle)
			sinCache[idx] = math.Sin(angle)
		}
		for head := 0; head < numHeads; head++ {
			offset := head * headDim
			for pair := 0; pair < half; pair++ {
				idx0 := offset + 2*pair
				idx1 := idx0 + 1
				x0 := row[idx0]
				x1 := row[idx1]
				c := cosCache[pair]
				s := sinCache[pair]
				row[idx0] = x0*c - x1*s
				row[idx1] = x0*s + x1*c
			}
		}
	}
}

func sliceHeadMatrix(m *util.Matrix64, headIdx, numHeads, headDim int) *util.Matrix64 {
	seqLen := m.Rows
	out := util.NewMatrix64(seqLen, headDim)
	for pos := 0; pos < seqLen; pos++ {
		row := m.Data[pos*m.Stride : pos*m.Stride+m.Cols]
		copy(out.Data[pos*out.Stride:pos*out.Stride+headDim], row[headIdx*headDim:(headIdx+1)*headDim])
	}
	return out
}

func cloneMatrix(m *util.Matrix64) *util.Matrix64 {
	if m == nil {
		return nil
	}
	clone := util.NewMatrix64(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		copy(clone.Data[i*clone.Stride:i*clone.Stride+m.Cols], m.Data[i*m.Stride:i*m.Stride+m.Cols])
	}
	return clone
}

func concatMatrixRows(a, b *util.Matrix64) *util.Matrix64 {
	if a == nil {
		return cloneMatrix(b)
	}
	if b == nil {
		return cloneMatrix(a)
	}
	if a.Cols != b.Cols {
		panic("concatMatrixRows: column mismatch")
	}
	out := util.NewMatrix64(a.Rows+b.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		copy(out.Data[i*out.Stride:i*out.Stride+a.Cols], a.Data[i*a.Stride:i*a.Stride+a.Cols])
	}
	for i := 0; i < b.Rows; i++ {
		dst := out.Data[(a.Rows+i)*out.Stride : (a.Rows+i)*out.Stride+a.Cols]
		src := b.Data[i*b.Stride : i*b.Stride+b.Cols]
		copy(dst, src)
	}
	return out
}

// Generate generates text from the model
func (vg *VaultGemma) Generate(prompt []int, maxTokens int) ([]int, error) {
	return vg.GenerateWithSampling(prompt, maxTokens, SamplingConfig{Temperature: 1})
}

func (vg *VaultGemma) GenerateWithSampling(prompt []int, maxTokens int, cfg SamplingConfig) ([]int, error) {
	if maxTokens <= 0 {
		return append([]int(nil), prompt...), nil
	}

	effectiveCfg := cfg
	if effectiveCfg.Temperature <= 0 {
		effectiveCfg.Temperature = 1
	}
	if effectiveCfg.TopP <= 0 || effectiveCfg.TopP > 1 {
		effectiveCfg.TopP = DefaultTopP
	}
	if effectiveCfg.TopK <= 0 {
		effectiveCfg.TopK = DefaultTopK
	}

	cache := make([]AttentionCache, vg.Config.NumLayers)

	logits, err := vg.Forward(prompt)
	if err != nil {
		return nil, err
	}
	for layerIndex := range vg.Layers {
		cache[layerIndex].Keys = nil
		cache[layerIndex].Values = nil
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	total := append([]int(nil), prompt...)

	for step := 0; step < maxTokens; step++ {
		lastLogits := logits.Data[(logits.Rows-1)*logits.Cols : logits.Rows*logits.Cols]
		nextToken := sampleLogits(lastLogits, effectiveCfg, rng)
		total = append(total, nextToken)

		if nextToken == vg.Config.EOSTokenID {
			break
		}

		nextHidden := vg.embedTokens([]int{nextToken})
		for layerIndex := range vg.Layers {
			layer := &vg.Layers[layerIndex]
			normed := vg.rmsNorm(nextHidden, layer.LayerNorm1)
			attnOut := vg.multiHeadAttentionWithCache(normed, layer.SelfAttention, &cache[layerIndex])
			residual := vg.addResidual(nextHidden, attnOut)
			normed = vg.rmsNorm(residual, layer.LayerNorm2)
			ffnOut := vg.feedForward(normed, layer.FeedForward)
			nextHidden = vg.addResidual(residual, ffnOut)
		}
		logits = vg.projectOutput(nextHidden)
	}

	return total, nil
}

func (vg *VaultGemma) argmax(logits []float64) int {
	maxIdx := 0
	maxVal := logits[0]

	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}

	return maxIdx
}
