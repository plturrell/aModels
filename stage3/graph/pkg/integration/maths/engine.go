package maths

import (
	"context"
	"fmt"
)

// Engine wraps the upstream maths Provider with context-aware helpers that the
// LangGraph runtime can depend on. All operations honour context cancellation
// and surface the Provider's errors directly.
type Engine struct {
	provider Provider
}

// NewEngine builds an Engine using the supplied Provider. When provider is nil,
// the default agenticAiETH maths backend is selected.
func NewEngine(provider Provider) *Engine {
	if provider == nil {
		provider = NewProvider()
	}
	return &Engine{provider: provider}
}

func (e *Engine) pre(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("maths engine: context is nil")
	}
	return ctx.Err()
}

func (e *Engine) post(ctx context.Context, err error) error {
	if err != nil {
		return err
	}
	return ctx.Err()
}

// Dot computes the dot product between two vectors.
func (e *Engine) Dot(ctx context.Context, a, b []float64) (float64, error) {
	if err := e.pre(ctx); err != nil {
		return 0, err
	}
	val := e.provider.Dot(a, b)
	if err := e.post(ctx, nil); err != nil {
		return 0, err
	}
	return val, nil
}

// Cosine calculates cosine similarity between two vectors.
func (e *Engine) Cosine(ctx context.Context, a, b []float64) (float64, error) {
	if err := e.pre(ctx); err != nil {
		return 0, err
	}
	val := e.provider.Cos(a, b)
	if err := e.post(ctx, nil); err != nil {
		return 0, err
	}
	return val, nil
}

// MatMul performs matrix multiplication of A (m x k) with B (k x n).
func (e *Engine) MatMul(ctx context.Context, m, n, k int, A, B []float64) ([]float64, error) {
	if err := e.pre(ctx); err != nil {
		return nil, err
	}
	out := e.provider.MatMul(m, n, k, A, B)
	if err := e.post(ctx, nil); err != nil {
		return nil, err
	}
	return out, nil
}

// Project multiplies matrix A (m x n) by projection P (n x r).
func (e *Engine) Project(ctx context.Context, m, n, r int, A, P []float64) ([]float64, error) {
	if err := e.pre(ctx); err != nil {
		return nil, err
	}
	out := e.provider.Project(m, n, r, A, P)
	if err := e.post(ctx, nil); err != nil {
		return nil, err
	}
	return out, nil
}

// CosineTopK returns the indices and scores of the top K vectors in A most
// similar to q, assuming row-major layout with vector dimension n.
func (e *Engine) CosineTopK(ctx context.Context, n int, A []float64, q []float64, topK int) ([]int, []float64, error) {
	if err := e.pre(ctx); err != nil {
		return nil, nil, err
	}
	indices, scores := e.provider.CosineTopK(n, A, q, topK)
	if err := e.post(ctx, nil); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

// CosineMultiTopK returns the top K matches for each query in Q.
func (e *Engine) CosineMultiTopK(ctx context.Context, n int, A []float64, Q []float64, topK int) ([][]int, [][]float64, error) {
	if err := e.pre(ctx); err != nil {
		return nil, nil, err
	}
	indices, scores := e.provider.CosineMultiTopK(n, A, Q, topK)
	if err := e.post(ctx, nil); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

// CosineTopKInt8 performs top-k similarity where documents are int8-quantised vectors.
func (e *Engine) CosineTopKInt8(ctx context.Context, n int, A8 []int8, q []float64, topK int) ([]int, []float64, error) {
	if err := e.pre(ctx); err != nil {
		return nil, nil, err
	}
	indices, scores := e.provider.CosineTopKInt8(n, A8, q, topK)
	if err := e.post(ctx, nil); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}
