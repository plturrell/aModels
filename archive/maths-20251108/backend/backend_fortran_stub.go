//go:build !cgo
// +build !cgo

package backend

// No-op: with cgo disabled, we keep the pure-Go provider as default.
// Users can force pure Go via INFRA_MATHS_BACKEND=go as well.
