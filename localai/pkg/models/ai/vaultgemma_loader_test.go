package ai

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

type tensorEntry struct {
	name  string
	shape []int
	data  []float32
	dtype string
}

func TestLoadVaultGemmaFromSafetensors_SmallModel(t *testing.T) {
	dir := t.TempDir()

	// Write minimal config
	cfg := VaultGemmaConfig{
		HiddenSize:       8,
		NumLayers:        1,
		NumHeads:         2,
		VocabSize:        4,
		MaxPositionEmbs:  32,
		IntermediateSize: 16,
		HeadDim:          4,
		RMSNormEps:       1e-5,
	}
	cfgBytes, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgBytes, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	// Prepare synthetic tensors
	entries := []tensorEntry{
		{
			name:  "model.embed_tokens.weight",
			shape: []int{4, 8},
			data: []float32{
				0, 1, 2, 3, 4, 5, 6, 7,
				1, 2, 3, 4, 5, 6, 7, 8,
				2, 3, 4, 5, 6, 7, 8, 9,
				3, 4, 5, 6, 7, 8, 9, 10,
			},
			dtype: "F32",
		},
		{
			name:  "model.layers.0.self_attn.q_proj.weight",
			shape: []int{8, 8},
			data:  makeFloatSequence(64),
			dtype: "F32",
		},
		{
			name:  "model.layers.0.mlp.up_proj.weight",
			shape: []int{8, 16},
			data:  makeFloatSequence(128),
			dtype: "F32",
		},
	}

	if err := writeSafetensors(filepath.Join(dir, "model.safetensors"), entries); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}

	model, err := LoadVaultGemmaFromSafetensors(dir)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	if got, want := model.Embed.Weights.Rows, 4; got != want {
		t.Fatalf("embed rows got %d want %d", got, want)
	}
	if got := model.Embed.Weights.Data[7]; got != 7 {
		t.Fatalf("embed data mismatch: got %f want %f", got, 7.0)
	}

	q := model.Layers[0].SelfAttention.WQ
	if q == nil || q.Rows != 8 || q.Cols != 8 {
		t.Fatalf("unexpected WQ shape %#v", q)
	}
	if q.Data[0] != 0 || q.Data[7] != 7 {
		t.Fatalf("unexpected WQ data: %#v", q.Data[:8])
	}

	up := model.Layers[0].FeedForward.W1
	if up == nil || up.Rows != 8 || up.Cols != 16 {
		t.Fatalf("unexpected W1 shape %#v", up)
	}
	if up.Data[0] != 0 || up.Data[10] != 10 {
		t.Fatalf("unexpected W1 data slice: %#v", up.Data[:16])
	}
}

func TestApplyRotaryEmbeddings(t *testing.T) {
	numHeads := 2
	headDim := 4
	seqLen := 3

	m := util.NewMatrix64(seqLen, numHeads*headDim)
	for i := range m.Data {
		m.Data[i] = float64(i + 1)
	}

	theta := 10000.0
	original := append([]float64(nil), m.Data...)
	applyRotaryEmbeddings(m, numHeads, headDim, theta)

	expected := util.NewMatrix64(seqLen, numHeads*headDim)
	copy(expected.Data, original)
	half := headDim / 2
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		exponent := float64(2*i) / float64(headDim)
		invFreq[i] = math.Pow(theta, -exponent)
	}
	for pos := 0; pos < seqLen; pos++ {
		for head := 0; head < numHeads; head++ {
			offset := head * headDim
			for pair := 0; pair < half; pair++ {
				idx0 := pos*expected.Stride + offset + 2*pair
				idx1 := idx0 + 1
				x0 := expected.Data[idx0]
				x1 := expected.Data[idx1]
				angle := float64(pos) * invFreq[pair]
				c := math.Cos(angle)
				s := math.Sin(angle)
				expected.Data[idx0] = x0*c - x1*s
				expected.Data[idx1] = x0*s + x1*c
			}
		}
	}

	for i := range m.Data {
		if math.Abs(m.Data[i]-expected.Data[i]) > 1e-12 {
			t.Fatalf("rotary mismatch at %d: got %.12f want %.12f", i, m.Data[i], expected.Data[i])
		}
	}
}

func TestMultiHeadAttentionMatchesFlashAttention(t *testing.T) {
	cfg := VaultGemmaConfig{
		HiddenSize:       4,
		NumLayers:        1,
		NumHeads:         2,
		VocabSize:        16,
		MaxPositionEmbs:  32,
		IntermediateSize: 8,
		HeadDim:          2,
		RMSNormEps:       1e-5,
		RopeTheta:        10000.0,
	}

	model := &VaultGemma{Config: cfg, Layers: make([]TransformerLayer, cfg.NumLayers)}
	model.initializeLayers()

	setIdentity := func(m *util.Matrix64) {
		for i := range m.Data {
			m.Data[i] = 0
		}
		for i := 0; i < m.Rows && i < m.Cols; i++ {
			m.Data[i*m.Stride+i] = 1
		}
	}

	setIdentity(model.Layers[0].SelfAttention.WQ)
	setIdentity(model.Layers[0].SelfAttention.WK)
	setIdentity(model.Layers[0].SelfAttention.WV)
	setIdentity(model.Layers[0].SelfAttention.WO)

	x := util.NewMatrix64(3, cfg.HiddenSize)
	for i := range x.Data {
		x.Data[i] = float64(i + 1)
	}

	output := model.multiHeadAttention(x, model.Layers[0].SelfAttention)

	QData := util.MatMul(x.Rows, cfg.NumHeads*cfg.HeadDim, x.Cols, x.Data, model.Layers[0].SelfAttention.WQ.Data)
	KData := util.MatMul(x.Rows, cfg.NumHeads*cfg.HeadDim, x.Cols, x.Data, model.Layers[0].SelfAttention.WK.Data)
	VData := util.MatMul(x.Rows, cfg.NumHeads*cfg.HeadDim, x.Cols, x.Data, model.Layers[0].SelfAttention.WV.Data)
	Q := &util.Matrix64{Data: QData, Rows: x.Rows, Cols: cfg.NumHeads * cfg.HeadDim, Stride: cfg.NumHeads * cfg.HeadDim}
	K := &util.Matrix64{Data: KData, Rows: x.Rows, Cols: cfg.NumHeads * cfg.HeadDim, Stride: cfg.NumHeads * cfg.HeadDim}
	V := &util.Matrix64{Data: VData, Rows: x.Rows, Cols: cfg.NumHeads * cfg.HeadDim, Stride: cfg.NumHeads * cfg.HeadDim}

	applyRotaryEmbeddings(Q, cfg.NumHeads, cfg.HeadDim, cfg.RopeTheta)
	applyRotaryEmbeddings(K, cfg.NumHeads, cfg.HeadDim, cfg.RopeTheta)

	concat := util.NewMatrix64(x.Rows, cfg.NumHeads*cfg.HeadDim)
	scale := 1.0 / math.Sqrt(float64(cfg.HeadDim))
	for h := 0; h < cfg.NumHeads; h++ {
		qHead := sliceHeadMatrix(Q, h, cfg.NumHeads, cfg.HeadDim)
		kHead := sliceHeadMatrix(K, h, cfg.NumHeads, cfg.HeadDim)
		vHead := sliceHeadMatrix(V, h, cfg.NumHeads, cfg.HeadDim)
		headOut := tensor.FlashAttention(qHead, kHead, vHead, scale)
		for pos := 0; pos < x.Rows; pos++ {
			dst := concat.Data[pos*concat.Stride+h*cfg.HeadDim : pos*concat.Stride+(h+1)*cfg.HeadDim]
			src := headOut.Data[pos*headOut.Stride : pos*headOut.Stride+cfg.HeadDim]
			copy(dst, src)
		}
	}

	expectedData := util.MatMul(concat.Rows, cfg.HiddenSize, concat.Cols, concat.Data, model.Layers[0].SelfAttention.WO.Data)
	expected := &util.Matrix64{Data: expectedData, Rows: concat.Rows, Cols: cfg.HiddenSize, Stride: cfg.HiddenSize}

	if len(output.Data) != len(expected.Data) {
		t.Fatalf("output size mismatch: got %d want %d", len(output.Data), len(expected.Data))
	}
	for i := range output.Data {
		if math.Abs(output.Data[i]-expected.Data[i]) > 1e-9 {
			t.Fatalf("attention mismatch at %d: got %.12f want %.12f", i, output.Data[i], expected.Data[i])
		}
	}
}

func TestAttentionCacheMatchesFullSequence(t *testing.T) {
	cfg := VaultGemmaConfig{
		HiddenSize:       4,
		NumLayers:        1,
		NumHeads:         2,
		VocabSize:        16,
		MaxPositionEmbs:  32,
		IntermediateSize: 8,
		HeadDim:          2,
		RMSNormEps:       1e-5,
		RopeTheta:        10000.0,
	}

	model := &VaultGemma{Config: cfg, Layers: make([]TransformerLayer, cfg.NumLayers)}
	model.initializeLayers()

	setIdentity := func(m *util.Matrix64) {
		for i := range m.Data {
			m.Data[i] = 0
		}
		for i := 0; i < m.Rows && i < m.Cols; i++ {
			m.Data[i*m.Stride+i] = 1
		}
	}

	setIdentity(model.Layers[0].SelfAttention.WQ)
	setIdentity(model.Layers[0].SelfAttention.WK)
	setIdentity(model.Layers[0].SelfAttention.WV)
	setIdentity(model.Layers[0].SelfAttention.WO)

	fullInput := []int{2, 5}
	fullHidden := model.embedTokens(fullInput)
	fullNorm := model.rmsNorm(fullHidden, model.Layers[0].LayerNorm1)
	fullOutput := model.multiHeadAttention(fullNorm, model.Layers[0].SelfAttention)
	want := fullOutput.Data[fullOutput.Stride : fullOutput.Stride+cfg.HiddenSize]

	cache := &AttentionCache{}

	firstHidden := model.embedTokens([]int{2})
	firstNorm := model.rmsNorm(firstHidden, model.Layers[0].LayerNorm1)
	model.multiHeadAttentionWithCache(firstNorm, model.Layers[0].SelfAttention, cache)

	secondHidden := model.embedTokens([]int{5})
	secondNorm := model.rmsNorm(secondHidden, model.Layers[0].LayerNorm1)
	cachedOutput := model.multiHeadAttentionWithCache(secondNorm, model.Layers[0].SelfAttention, cache)

	got := cachedOutput.Data[:cfg.HiddenSize]
	for i := 0; i < cfg.HiddenSize; i++ {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("cache mismatch at %d: got %.12f want %.12f", i, got[i], want[i])
		}
	}
}

func BenchmarkReadFloat32Tensor(b *testing.B) {
	const elements = 1 << 20 // ~4MB of float32 data

	file, err := os.CreateTemp("", "float32-*.bin")
	if err != nil {
		b.Fatalf("create temp: %v", err)
	}
	defer os.Remove(file.Name())
	defer file.Close()

	buf := make([]byte, elements*float32Size)
	for i := 0; i < elements; i++ {
		binary.LittleEndian.PutUint32(buf[i*float32Size:(i+1)*float32Size], math.Float32bits(float32(i)))
	}
	if _, err := file.Write(buf); err != nil {
		b.Fatalf("write temp: %v", err)
	}

	model := &VaultGemma{}
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := file.Seek(0, io.SeekStart); err != nil {
			b.Fatalf("seek: %v", err)
		}
		if _, err := model.readFloat32Tensor(file, elements); err != nil {
			b.Fatalf("read tensor: %v", err)
		}
	}
}

func writeSafetensors(path string, tensors []tensorEntry) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	header := make(map[string]TensorInfo, len(tensors))
	var offset int64
	for _, entry := range tensors {
		bytes := len(entry.data) * float32Size
		header[entry.name] = TensorInfo{
			DType:       entry.dtype,
			Shape:       entry.shape,
			DataOffsets: [2]int64{offset, offset + int64(bytes)},
		}
		offset += int64(bytes)
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}

	if err := binary.Write(f, binary.LittleEndian, int64(len(headerBytes))); err != nil {
		return err
	}
	if _, err := f.Write(headerBytes); err != nil {
		return err
	}

	for _, entry := range tensors {
		for _, value := range entry.data {
			if err := binary.Write(f, binary.LittleEndian, value); err != nil {
				return err
			}
		}
	}

	return nil
}

func makeFloatSequence(n int) []float32 {
	seq := make([]float32, n)
	for i := range seq {
		seq[i] = float32(i)
	}
	return seq
}
