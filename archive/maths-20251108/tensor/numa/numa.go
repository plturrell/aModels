package numa

import (
    "fmt"
    "math"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
)

type ExtremeConfig struct {
    UseAVX512       bool
    UseFMA          bool
    UseNUMA         bool
    UseJIT          bool
    UseQuantization bool
    NumSockets      int
    CoresPerSocket  int
}

func DetectExtremeCapabilities() *ExtremeConfig {
    return &ExtremeConfig{UseFMA: true, UseQuantization: true, NumSockets: 1, CoresPerSocket: 0}
}

// min moved to internal/ints

// AVX512MatMul: simple cache-blocked matmul (portable fallback)
func AVX512MatMul(A, B [][]float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 { return nil, ErrDim() }
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n) }
    const blk = 128
    for ii := 0; ii < m; ii += blk {
        iEnd := ints.Min(ii+blk, m)
        for jj := 0; jj < n; jj += blk {
            jEnd := ints.Min(jj+blk, n)
            for kk := 0; kk < k1; kk += blk {
                kEnd := ints.Min(kk+blk, k1)
                for i := ii; i < iEnd; i++ {
                    for j := jj; j < jEnd; j++ {
                        s := C[i][j]
                        for p := kk; p < kEnd; p++ { s += A[i][p]*B[p][j] }
                        C[i][j] = s
                    }
                }
            }
        }
    }
    return C, nil
}

type NUMAMatrix struct { data [][]float64; sockets int; rowsPerSocket int }

func NewNUMAMatrix(m, n, sockets int) *NUMAMatrix {
    if sockets <= 0 { sockets = 1 }
    rps := (m + sockets - 1) / sockets
    nm := &NUMAMatrix{ data: make([][]float64, m), sockets: sockets, rowsPerSocket: rps }
    for i := 0; i < m; i++ { nm.data[i] = make([]float64, n) }
    return nm
}

func NUMAMatMul(A, B *NUMAMatrix) (*NUMAMatrix, error) {
    m := len(A.data); k := len(A.data[0]); n := len(B.data[0])
    C := NewNUMAMatrix(m, n, A.sockets)
    for i := 0; i < m; i++ { for j := 0; j < n; j++ { s:=0.0; for p:=0;p<k;p++{ s+=A.data[i][p]*B.data[p][j] }; C.data[i][j]=s } }
    return C, nil
}

func FusedConvBNReLU(input, kernel [][]float64, gamma, beta []float64, eps float64) ([][]float64, error) {
    inH,inW := len(input), len(input[0]); kH,kW := len(kernel), len(kernel[0])
    outH := inH - kH + 1; outW := inW - kW + 1
    out := make([][]float64, outH); for i:=0;i<outH;i++{ out[i]=make([]float64,outW) }
    for i:=0;i<outH;i++{ for j:=0;j<outW;j++{ sum:=0.0; for ki:=0;ki<kH;ki++{ for kj:=0;kj<kW;kj++{ sum += input[i+ki][j+kj]*kernel[ki][kj] } }; norm := (sum-0)/math.Sqrt(1+eps); v := gamma[0]*norm + beta[0]; if v<0 { v=0 }; out[i][j]=v } }
    return out,nil
}

func FusedLinearLayerForward(input, weight [][]float64, bias []float64, activation string) ([][]float64, error) {
    m,k := len(input), len(input[0]); n := len(weight[0])
    out := make([][]float64,m); for i:=0;i<m;i++{ out[i]=make([]float64,n) }
    const blk = 64
    for ii:=0; ii<m; ii+=blk { iEnd:=ints.Min(ii+blk,m); for jj:=0;jj<n;jj+=blk{ jEnd:=ints.Min(jj+blk,n); for kk:=0; kk<k; kk+=blk{ kEnd:=ints.Min(kk+blk,k); for i:=ii;i<iEnd;i++{ for j:=jj;j<jEnd;j++{ s:=out[i][j]; for p:=kk;p<kEnd;p++{ s += input[i][p]*weight[p][j] }; out[i][j]=s } } } } }
    for i:=0;i<m;i++{ for j:=0;j<n;j++{ v:=out[i][j]+bias[j]; switch activation{ case "relu": if v<0 { v=0 }; case "gelu": v=0.5*v*(1+math.Tanh(0.7978845608*(v+0.044715*v*v*v))); case "silu": v = v/(1+math.Exp(-v)) }; out[i][j]=v } }
    return out,nil
}

func ErrDim() error { return fmt.Errorf("dimension mismatch") }
