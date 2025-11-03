package activations

import "testing"

func benchArrayF64(n int) []float64 { a:=make([]float64,n); for i:=range a{ a[i]=float64(i%97)/33.0 }; return a }
func benchArrayF32(n int) []float32 { a:=make([]float32,n); for i:=range a{ a[i]=float32(i%97)/33.0 }; return a }

func BenchmarkVectorizedTanhF64(b *testing.B) { x:=benchArrayF64(1<<16); b.ResetTimer(); for i:=0;i<b.N;i++{ _ = VectorizedTanhF64(x) } }
func BenchmarkVectorizedLog1pF64(b *testing.B){ x:=benchArrayF64(1<<16); b.ResetTimer(); for i:=0;i<b.N;i++{ _ = VectorizedLog1pF64(x) } }
func BenchmarkVectorizedExpm1F64(b *testing.B){ x:=benchArrayF64(1<<16); b.ResetTimer(); for i:=0;i<b.N;i++{ _ = VectorizedExpm1F64(x) } }
func BenchmarkVectorizedTanhF32(b *testing.B) { x:=benchArrayF32(1<<17); b.ResetTimer(); for i:=0;i<b.N;i++{ _ = VectorizedTanhF32(x) } }

