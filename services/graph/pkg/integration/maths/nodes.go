package maths

import (
	"context"
	"errors"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// DotInput represents the payload required for a dot product operation.
type DotInput struct {
	A []float64 `json:"a"`
	B []float64 `json:"b"`
}

// DotNode returns a graph node handler that computes the dot product.
func DotNode(engine *Engine) stategraph.NodeFunc {
	if engine == nil {
		engine = NewEngine(nil)
	}
	return func(ctx context.Context, input any) (any, error) {
		payload, ok := input.(DotInput)
		if !ok {
			return nil, errors.New("maths dot node: input must be maths.DotInput")
		}
		value, err := engine.Dot(ctx, payload.A, payload.B)
		if err != nil {
			return nil, err
		}
		return value, nil
	}
}

// CosineTopKInput represents the payload for cosine Top-K search.
type CosineTopKInput struct {
	Dimension int
	Matrix    []float64
	Query     []float64
	TopK      int
}

// CosineTopKResult captures cosine Top-K outputs.
type CosineTopKResult struct {
	Indices []int     `json:"indices"`
	Scores  []float64 `json:"scores"`
}

// CosineTopKNode returns a node handler performing cosine Top-K.
func CosineTopKNode(engine *Engine) stategraph.NodeFunc {
	if engine == nil {
		engine = NewEngine(nil)
	}
	return func(ctx context.Context, input any) (any, error) {
		payload, ok := input.(CosineTopKInput)
		if !ok {
			return nil, errors.New("maths cosine node: input must be maths.CosineTopKInput")
		}
		idx, scores, err := engine.CosineTopK(ctx, payload.Dimension, payload.Matrix, payload.Query, payload.TopK)
		if err != nil {
			return nil, err
		}
		return CosineTopKResult{Indices: idx, Scores: scores}, nil
	}
}

// MatMulInput represents the payload for matrix multiplication.
type MatMulInput struct {
	M int
	N int
	K int
	A []float64
	B []float64
}

// MatMulNode returns a node handler that computes matrix multiplication.
func MatMulNode(engine *Engine) stategraph.NodeFunc {
	if engine == nil {
		engine = NewEngine(nil)
	}
	return func(ctx context.Context, input any) (any, error) {
		payload, ok := input.(MatMulInput)
		if !ok {
			return nil, errors.New("maths matmul node: input must be maths.MatMulInput")
		}
		out, err := engine.MatMul(ctx, payload.M, payload.N, payload.K, payload.A, payload.B)
		if err != nil {
			return nil, err
		}
		return out, nil
	}
}
