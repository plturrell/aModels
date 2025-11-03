package simd

import (
	"runtime"

	cpu "golang.org/x/sys/cpu"
)

// HasAVX512 reports whether the current platform supports AVX-512 instructions.
func HasAVX512() bool {
	if runtime.GOARCH != "amd64" {
		return false
	}
	return cpu.X86.HasAVX512F
}

// HasAVX2 reports whether the current platform supports AVX2 instructions.
func HasAVX2() bool {
	if runtime.GOARCH != "amd64" {
		return false
	}
	return cpu.X86.HasAVX2
}

// HasNEON reports whether the current platform provides NEON (ARM) SIMD.
func HasNEON() bool {
	return runtime.GOARCH == "arm64"
}

// HasSSE4 reports whether the current platform supports SSE4.2.
func HasSSE4() bool {
	if runtime.GOARCH != "amd64" {
		return false
	}
	return cpu.X86.HasSSE42
}
