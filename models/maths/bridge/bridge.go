package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unsafe"

	maths "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths"
)

const (
	minInt8 = -128
	maxInt8 = 127
)

type request struct {
	Operation string          `json:"operation"`
	Payload   json.RawMessage `json:"payload"`
}

type response struct {
	Result interface{} `json:"result,omitempty"`
	Error  string      `json:"error,omitempty"`
}

type handler func(json.RawMessage) (interface{}, error)

var operationHandlers map[string]handler
var operationList []string

func init() {
	operationHandlers = map[string]handler{
		"abs":                         handleAbs,
		"abs_int":                     handleAbsInt,
		"add":                         handleAdd,
		"add_int":                     handleAddInt,
		"build_random_projection":     handleBuildRandomProjection,
		"ceil":                        handleCeil,
		"clear_cache":                 handleClearCache,
		"cosine_auto":                 handleCosineAuto,
		"cosine_batch_auto":           handleCosineBatchAuto,
		"cosine_multi_top_k":          handleCosineMultiTopK,
		"cosine_top_k":                handleCosineTopK,
		"cosine_top_k_int8":           handleCosineTopKInt8,
		"divide":                      handleDivide,
		"divide_int":                  handleDivideInt,
		"dot":                         handleDot,
		"dot_auto":                    handleDotAuto,
		"dot_batch_auto":              handleDotBatchAuto,
		"equal":                       handleEqual,
		"equal_int":                   handleEqualInt,
		"expm1_estrin_f64":            handleExpm1EstrinF64,
		"flash_attention_2d_f32":      handleFlashAttention2D32,
		"floor":                       handleFloor,
		"fused_add_mul_exp":           handleFusedAddMulExp,
		"fused_softmax_cross_entropy": handleFusedSoftmaxCrossEntropy,
		"get_active_simd_path":        handleGetActiveSIMDPath,
		"get_bottlenecks":             handleGetBottlenecks,
		"get_cache_stats":             handleGetCacheStats,
		"get_operation_heatmap":       handleGetOperationHeatmap,
		"get_performance_metrics":     handleGetPerformanceMetrics,
		"get_simd_accuracy_modes":     handleGetSIMDAccuracyModes,
		"get_simd_capabilities":       handleGetSIMDCapabilities,
		"get_top_operations":          handleGetTopOperations,
		"greater":                     handleGreater,
		"greater_int":                 handleGreaterInt,
		"log1p_estrin_f64":            handleLog1pEstrinF64,
		"matmul":                      handleMatMul,
		"matmul2d":                    handleMatMul2D,
		"max":                         handleMax,
		"max_int":                     handleMaxInt,
		"mean":                        handleMean,
		"min":                         handleMin,
		"min_int":                     handleMinInt,
		"modulo":                      handleModulo,
		"modulo_int":                  handleModuloInt,
		"multiply":                    handleMultiply,
		"multiply_int":                handleMultiplyInt,
		"parse_float":                 handleParseFloat,
		"parse_int":                   handleParseInt,
		"project":                     handleProject,
		"q16_from_float":              handleQ16FromFloat,
		"q16_to_float":                handleQ16ToFloat,
		"round":                       handleRound,
		"set_accel_config":            handleSetAccelConfig,
		"set_cache_size":              handleSetCacheSize,
		"set_simd_accuracy_mode":      handleSetSIMDAccuracyMode,
		"simd_fused_multiply_add_f32": handleSIMDFusedMultiplyAddF32,
		"sqrt":                        handleSqrt,
		"subtract":                    handleSubtract,
		"subtract_int":                handleSubtractInt,
		"sum":                         handleSum,
		"sum_int":                     handleSumInt,
		"tanh_estrin_f32":             handleTanhEstrinF32,
		"tanh_estrin_f64":             handleTanhEstrinF64,
		"vectorized_expm1_f64":        handleVectorizedExpm1F64,
		"vectorized_log1p_f64":        handleVectorizedLog1pF64,
		"vectorized_tanh_f32":         handleVectorizedTanhF32,
		"vectorized_tanh_f64":         handleVectorizedTanhF64,
		"softmax_2d_f32":              handleSoftmax2D32,
		"softmax_2d_f64":              handleSoftmax2D64,
		"softmax_row_f32":             handleSoftmaxRow32,
		"softmax_row_f64":             handleSoftmaxRow64,
		"softmax_cross_entropy_f32":   handleSoftmaxCrossEntropy32,
	}

	operationList = make([]string, 0, len(operationHandlers))
	for name := range operationHandlers {
		operationList = append(operationList, name)
	}
	sort.Strings(operationList)
}

func makeResponse(result interface{}, err error) *C.char {
	resp := response{Result: result}
	if err != nil {
		resp.Result = nil
		resp.Error = err.Error()
	}
	data, marshalErr := json.Marshal(resp)
	if marshalErr != nil {
		fallback, _ := json.Marshal(response{Error: fmt.Sprintf("failed to marshal response: %v", marshalErr)})
		return C.CString(string(fallback))
	}
	return C.CString(string(data))
}

func decodePayload(payload json.RawMessage, dst interface{}) error {
	if len(payload) == 0 || string(payload) == "null" || string(payload) == "" {
		payload = []byte("{}")
	}
	return json.Unmarshal(payload, dst)
}

//export ExecuteMaths
func ExecuteMaths(cInput *C.char) *C.char {
	if cInput == nil {
		return makeResponse(nil, errors.New("nil request"))
	}

	input := C.GoString(cInput)
	if strings.TrimSpace(input) == "" {
		return makeResponse(nil, errors.New("empty request"))
	}

	var req request
	if err := json.Unmarshal([]byte(input), &req); err != nil {
		return makeResponse(nil, fmt.Errorf("invalid request: %w", err))
	}

	if req.Operation == "" {
		return makeResponse(nil, errors.New("missing operation"))
	}

	if req.Operation == "__list_operations" {
		return makeResponse(operationList, nil)
	}

	handler, ok := operationHandlers[req.Operation]
	if !ok {
		return makeResponse(nil, fmt.Errorf("unknown operation %q", req.Operation))
	}

	result, err := handler(req.Payload)
	return makeResponse(result, err)
}

//export FreeCString
func FreeCString(ptr *C.char) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

func handleMatMul2D(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A [][]float64 `json:"a"`
		B [][]float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.MatMul2D(args.A, args.B), nil
}

func handleMatMul(payload json.RawMessage) (interface{}, error) {
	var args struct {
		M int       `json:"m"`
		N int       `json:"n"`
		K int       `json:"k"`
		A []float64 `json:"a"`
		B []float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.MatMul(args.M, args.N, args.K, args.A, args.B), nil
}

func handleSetSIMDAccuracyMode(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Mode string `json:"mode"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}

	mode := strings.ToLower(strings.TrimSpace(args.Mode))
	switch mode {
	case "", "fast", "fast_approximate":
		maths.SetSIMDAccuracyMode(maths.SIMDFastApproximate)
	case "strict", "strict_accuracy":
		maths.SetSIMDAccuracyMode(maths.SIMDStrictAccuracy)
	default:
		return nil, fmt.Errorf("unsupported SIMD accuracy mode %q", args.Mode)
	}
	return map[string]string{"status": "ok"}, nil
}

func handleTanhEstrinF64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.TanhEstrinF64(args.X), nil
}

func handleLog1pEstrinF64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Log1pEstrinF64(args.X), nil
}

func handleExpm1EstrinF64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Expm1EstrinF64(args.X), nil
}

func handleTanhEstrinF32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float32 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.TanhEstrinF32(args.X), nil
}

func handleVectorizedTanhF64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X []float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.VectorizedTanhF64(args.X), nil
}

func handleVectorizedLog1pF64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X []float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.VectorizedLog1pF64(args.X), nil
}

func handleVectorizedExpm1F64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X []float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.VectorizedExpm1F64(args.X), nil
}

func handleVectorizedTanhF32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X []float32 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.VectorizedTanhF32(args.X), nil
}

func handleFusedSoftmaxCrossEntropy(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Logits [][]float64 `json:"logits"`
		Labels []int       `json:"labels"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	loss, probs := maths.FusedSoftmaxCrossEntropy(args.Logits, args.Labels)
	return map[string]interface{}{
		"loss":          loss,
		"probabilities": probs,
	}, nil
}

func handleSoftmaxRow32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X []float32 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.SoftmaxRow32(args.X), nil
}

func handleSoftmax2D32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X [][]float32 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Softmax2D32(args.X), nil
}

func handleSoftmaxCrossEntropy32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Logits [][]float32 `json:"logits"`
		Labels []int       `json:"labels"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	loss, probs := maths.SoftmaxCrossEntropy32(args.Logits, args.Labels)
	return map[string]interface{}{
		"loss":          loss,
		"probabilities": probs,
	}, nil
}

func handleSoftmaxRow64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X []float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.SoftmaxRow64(args.X), nil
}

func handleSoftmax2D64(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X [][]float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Softmax2D64(args.X), nil
}

func handleFlashAttention2D32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Q     [][]float32 `json:"q"`
		K     [][]float32 `json:"k"`
		V     [][]float32 `json:"v"`
		Scale float32     `json:"scale"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.FlashAttention2D32(args.Q, args.K, args.V, args.Scale), nil
}

func handleSIMDFusedMultiplyAddF32(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A []float32 `json:"a"`
		B []float32 `json:"b"`
		C []float32 `json:"c"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.SIMDFusedMultiplyAddF32(args.A, args.B, args.C), nil
}

func handleBuildRandomProjection(payload json.RawMessage) (interface{}, error) {
	var args struct {
		N    int   `json:"n"`
		R    int   `json:"r"`
		Seed int64 `json:"seed"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.BuildRandomProjection(args.N, args.R, args.Seed), nil
}

func handleProject(payload json.RawMessage) (interface{}, error) {
	var args struct {
		M int       `json:"m"`
		N int       `json:"n"`
		R int       `json:"r"`
		A []float64 `json:"a"`
		P []float64 `json:"p"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Project(args.M, args.N, args.R, args.A, args.P), nil
}

func handleCosineTopK(payload json.RawMessage) (interface{}, error) {
	var args struct {
		N    int       `json:"n"`
		A    []float64 `json:"a"`
		Q    []float64 `json:"q"`
		TopK int       `json:"top_k"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	indices, scores := maths.CosineTopK(args.N, args.A, args.Q, args.TopK)
	return map[string]interface{}{
		"indices": indices,
		"scores":  scores,
	}, nil
}

func handleCosineMultiTopK(payload json.RawMessage) (interface{}, error) {
	var args struct {
		N    int       `json:"n"`
		A    []float64 `json:"a"`
		Q    []float64 `json:"q"`
		TopK int       `json:"top_k"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	indices, scores := maths.CosineMultiTopK(args.N, args.A, args.Q, args.TopK)
	return map[string]interface{}{
		"indices": indices,
		"scores":  scores,
	}, nil
}

func handleCosineTopKInt8(payload json.RawMessage) (interface{}, error) {
	var args struct {
		N    int       `json:"n"`
		A    []int     `json:"a"`
		Q    []float64 `json:"q"`
		TopK int       `json:"top_k"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	a8 := make([]int8, len(args.A))
	for i, v := range args.A {
		if v < minInt8 || v > maxInt8 {
			return nil, fmt.Errorf("value %d at index %d out of int8 range", v, i)
		}
		a8[i] = int8(v)
	}
	indices, scores := maths.CosineTopKInt8(args.N, a8, args.Q, args.TopK)
	return map[string]interface{}{
		"indices": indices,
		"scores":  scores,
	}, nil
}

func handleCosineAuto(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A []float64 `json:"a"`
		B []float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.CosineAuto(args.A, args.B), nil
}

func handleCosineBatchAuto(payload json.RawMessage) (interface{}, error) {
	var args struct {
		N int       `json:"n"`
		A []float64 `json:"a"`
		B []float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.CosineBatchAuto(args.N, args.A, args.B), nil
}

func handleDotAuto(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A []float64 `json:"a"`
		B []float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.DotAuto(args.A, args.B), nil
}

func handleDotBatchAuto(payload json.RawMessage) (interface{}, error) {
	var args struct {
		N int       `json:"n"`
		A []float64 `json:"a"`
		B []float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.DotBatchAuto(args.N, args.A, args.B), nil
}

func handleDot(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A []float64 `json:"a"`
		B []float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Dot(args.A, args.B), nil
}

func handleQ16FromFloat(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Q16FromFloat(args.X), nil
}

func handleQ16ToFloat(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Q int32 `json:"q"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Q16ToFloat(args.Q), nil
}

func handleParseInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Value string `json:"value"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.ParseInt(args.Value)
}

func handleParseFloat(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Value string `json:"value"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.ParseFloat(args.Value)
}

func handleSetAccelConfig(payload json.RawMessage) (interface{}, error) {
	var args struct {
		EnableFortran bool `json:"enable_fortran"`
		Threads       int  `json:"threads"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	maths.SetAccelConfig(maths.AccelConfig{
		EnableFortran: args.EnableFortran,
		Threads:       args.Threads,
	})
	return map[string]string{"status": "ok"}, nil
}

func handleFusedAddMulExp(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A []float64 `json:"a"`
		B []float64 `json:"b"`
		C []float64 `json:"c"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.FusedAddMulExp(args.A, args.B, args.C), nil
}

func handleSqrt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Sqrt(args.X), nil
}

func handleAdd(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Add(args.A, args.B), nil
}

func handleSubtract(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Subtract(args.A, args.B), nil
}

func handleMultiply(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Multiply(args.A, args.B), nil
}

func handleDivide(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Divide(args.A, args.B), nil
}

func handleModulo(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Modulo(args.A, args.B), nil
}

func handleAbs(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Abs(args.X), nil
}

func handleEqual(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Equal(args.A, args.B), nil
}

func handleGreater(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Greater(args.A, args.B), nil
}

func handleLess(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Less(args.A, args.B), nil
}

func handleRound(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Round(args.X), nil
}

func handleFloor(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Floor(args.X), nil
}

func handleCeil(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X float64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Ceil(args.X), nil
}

func handleSum(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []float64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Sum(args.Values), nil
}

func handleMin(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []float64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Min(args.Values), nil
}

func handleMax(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []float64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Max(args.Values), nil
}

func handleMean(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []float64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.Mean(args.Values), nil
}

func handleAddInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.AddInt(args.A, args.B), nil
}

func handleSubtractInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.SubtractInt(args.A, args.B), nil
}

func handleMultiplyInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.MultiplyInt(args.A, args.B), nil
}

func handleDivideInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	if args.B == 0 {
		return nil, errors.New("division by zero")
	}
	return maths.DivideInt(args.A, args.B), nil
}

func handleModuloInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	if args.B == 0 {
		return nil, errors.New("division by zero")
	}
	return maths.ModuloInt(args.A, args.B), nil
}

func handleAbsInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		X int64 `json:"x"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.AbsInt(args.X), nil
}

func handleEqualInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.EqualInt(args.A, args.B), nil
}

func handleGreaterInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.GreaterInt(args.A, args.B), nil
}

func handleLessInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		A int64 `json:"a"`
		B int64 `json:"b"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.LessInt(args.A, args.B), nil
}

func handleSumInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []int64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.SumInt(args.Values), nil
}

func handleMinInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []int64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.MinInt(args.Values), nil
}

func handleMaxInt(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Values []int64 `json:"values"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	return maths.MaxInt(args.Values), nil
}

func handleGetPerformanceMetrics(json.RawMessage) (interface{}, error) {
	return maths.GetPerformanceMetrics(), nil
}

func handleGetCacheStats(json.RawMessage) (interface{}, error) {
	return maths.GetCacheStats(), nil
}

func handleClearCache(json.RawMessage) (interface{}, error) {
	maths.ClearCache()
	return map[string]string{"status": "ok"}, nil
}

func handleSetCacheSize(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Size int `json:"size"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	if args.Size <= 0 {
		return nil, fmt.Errorf("cache size must be positive (got %d)", args.Size)
	}
	maths.SetCacheSize(args.Size)
	return map[string]string{"status": "ok"}, nil
}

func handleGetTopOperations(payload json.RawMessage) (interface{}, error) {
	var args struct {
		Limit int `json:"limit"`
	}
	if err := decodePayload(payload, &args); err != nil {
		return nil, err
	}
	if args.Limit <= 0 {
		args.Limit = 10
	}
	return maths.GetTopOperations(args.Limit), nil
}

func handleGetBottlenecks(json.RawMessage) (interface{}, error) {
	return maths.GetBottlenecks(), nil
}

func handleGetOperationHeatmap(json.RawMessage) (interface{}, error) {
	return maths.GetOperationHeatmap(), nil
}

func handleGetActiveSIMDPath(json.RawMessage) (interface{}, error) {
	return maths.GetActiveSIMDPath(), nil
}

func handleGetSIMDCapabilities(json.RawMessage) (interface{}, error) {
	return maths.GetSIMDCapabilities(), nil
}

func handleGetSIMDAccuracyModes(json.RawMessage) (interface{}, error) {
	return map[string]int{
		"fast_approximate": int(maths.SIMDFastApproximate),
		"strict_accuracy":  int(maths.SIMDStrictAccuracy),
	}, nil
}

func main() {}
