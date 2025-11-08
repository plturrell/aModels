package ops

import (
    "fmt"
    "strings"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
    gemm "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/gemm"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// Public operations (package-level) used by tensor wrappers

func Einsum(subscripts string, operands ...interface{}) (interface{}, error) {
    parts := strings.Split(subscripts, "->")
    if len(parts) != 2 { return nil, fmt.Errorf("invalid einsum notation: %s", subscripts) }
    inputs := strings.Split(parts[0], ",")
    output := strings.TrimSpace(parts[1])
    if len(inputs) != len(operands) { return nil, fmt.Errorf("number of operands (%d) doesn't match subscripts (%d)", len(operands), len(inputs)) }

    if len(inputs) == 2 {
        if inputs[0] == "ij" && inputs[1] == "jk" && output == "ik" {
            A, ok1 := operands[0].([][]float64); B, ok2 := operands[1].([][]float64)
            if !ok1 || !ok2 { return nil, fmt.Errorf("operands must be [][]float64 for matmul") }
            m, k1 := len(A), len(A[0]); k2, n := len(B), len(B[0])
            if k1 != k2 { return nil, fmt.Errorf("dimension mismatch: %d != %d", k1, k2) }
            return matmul(A, B, m, n, k1), nil
        }
        if inputs[0] == "bij" && inputs[1] == "bjk" && output == "bik" {
            A, ok1 := operands[0].([][][]float64); B, ok2 := operands[1].([][][]float64)
            if !ok1 || !ok2 { return nil, fmt.Errorf("operands must be [][][]float64 for batch matmul") }
            // Simple per-batch matmul
            m := len(A[0]); k := len(A[0][0]); n := len(B[0][0])
            out := make([][][]float64, len(A))
            for b := range A { out[b] = matmul(A[b], B[b], m, n, k) }
            return out, nil
        }
        if inputs[0] == "i" && inputs[1] == "j" && output == "ij" {
            a, ok1 := operands[0].([]float64); b, ok2 := operands[1].([]float64)
            if !ok1 || !ok2 { return nil, fmt.Errorf("operands must be []float64 for outer product") }
            return outerProduct(a, b), nil
        }
        if inputs[0] == "ij" && inputs[1] == "ij" && output == "ij" {
            A, ok1 := operands[0].([][]float64); B, ok2 := operands[1].([][]float64)
            if !ok1 || !ok2 { return nil, fmt.Errorf("operands must be [][]float64") }
            return elementwiseMul(A, B)
        }
    }

    if len(inputs) == 1 {
        if inputs[0] == "ii" && output == "" {
            A, ok := operands[0].([][]float64); if !ok { return nil, fmt.Errorf("operand must be [][]float64 for trace") }
            return trace(A), nil
        }
        if inputs[0] == "ij" && output == "ji" {
            A, ok := operands[0].([][]float64); if !ok { return nil, fmt.Errorf("operand must be [][]float64 for transpose") }
            return Transpose(A), nil
        }
        if inputs[0] == "ij" && output == "i" {
            A, ok := operands[0].([][]float64); if !ok { return nil, fmt.Errorf("operand must be [][]float64") }
            return sumAxis1(A), nil
        }
        if inputs[0] == "ij" && output == "j" {
            A, ok := operands[0].([][]float64); if !ok { return nil, fmt.Errorf("operand must be [][]float64") }
            return sumAxis0(A), nil
        }
    }
    return nil, fmt.Errorf("unsupported einsum pattern: %s", subscripts)
}

func Transpose(A [][]float64) [][]float64 { m,n := len(A), len(A[0]); R:=make([][]float64,n); for i:=0;i<n;i++{R[i]=make([]float64,m); for j:=0;j<m;j++{R[i][j]=A[j][i]}}; return R }

func Reshape(data []float64, shape ...int) (interface{}, error) {
    total := 1; for _,d:= range shape { if d<=0 { return nil, fmt.Errorf("invalid dimension: %d", d) }; total*=d }
    if total != len(data) { return nil, fmt.Errorf("cannot reshape array of size %d into shape %v", len(data), shape) }
    switch len(shape) { case 1: return data, nil; case 2: return reshape2D(data, shape[0], shape[1]), nil; case 3: return reshape3D(data, shape[0], shape[1], shape[2]), nil }
    return nil, fmt.Errorf("reshape only supports up to 3D")
}

func Broadcast(A, B [][]float64) ([][]float64, [][]float64, error) {
    m1,n1 := len(A), len(A[0]); m2,n2 := len(B), len(B[0])
    m := ints.Max(m1,m2); n := ints.Max(n1,n2)
    if (m1!=1 && m1!=m) || (n1!=1 && n1!=n) { return nil, nil, fmt.Errorf("cannot broadcast shape (%d,%d) to (%d,%d)", m1,n1,m,n) }
    if (m2!=1 && m2!=m) || (n2!=1 && n2!=n) { return nil, nil, fmt.Errorf("cannot broadcast shape (%d,%d) to (%d,%d)", m2,n2,m,n) }
    aOut := make([][]float64, m); bOut := make([][]float64, m)
    for i:=0;i<m;i++{ aOut[i]=make([]float64,n); bOut[i]=make([]float64,n); for j:=0;j<n;j++{ iA:=i; if m1==1 { iA=0 }; jA:=j; if n1==1 { jA=0 }; iB:=i; if m2==1 { iB=0 }; jB:=j; if n2==1 { jB=0 }; aOut[i][j]=A[iA][jA]; bOut[i][j]=B[iB][jB] } }
    return aOut,bOut,nil
}

func Permute(A [][][]float64, dims []int) ([][][]float64, error) {
    if len(dims) != 3 { return nil, fmt.Errorf("permute requires 3 dimensions for 3D tensor") }
    seen:=map[int]bool{}; for _,d:=range dims{ if d<0||d>2 {return nil, fmt.Errorf("invalid dimension: %d", d)}; if seen[d]{return nil, fmt.Errorf("duplicate dimension: %d", d)}; seen[d]=true }
    shape := []int{len(A), len(A[0]), len(A[0][0])}
    newShape := []int{shape[dims[0]], shape[dims[1]], shape[dims[2]]}
    R := make([][][]float64, newShape[0]); for i:=0;i<newShape[0];i++{ R[i]=make([][]float64,newShape[1]); for j:=0;j<newShape[1];j++{ R[i][j]=make([]float64,newShape[2]) } }
    for i:=0;i<shape[0];i++{ for j:=0;j<shape[1];j++{ for k:=0;k<shape[2];k++{ idx:=[]int{i,j,k}; R[idx[dims[0]]][idx[dims[1]]][idx[dims[2]]] = A[i][j][k] } } }
    return R,nil
}

// helpers
func matmul(A, B [][]float64, m, n, k int) [][]float64 { a:=util.From2D(A); b:=util.From2D(B); c:=gemm.MatMulContiguous(a,b); return util.To2D(c) }
func trace(A [][]float64) float64 { s:=0.0; for i:=0;i<len(A)&&i<len(A[0]);i++{ s+=A[i][i] }; return s }
func sumAxis1(A [][]float64) []float64 { m:=len(A); out:=make([]float64,m); for i:=0;i<m;i++{ s:=0.0; for j:=0;j<len(A[i]);j++{ s+=A[i][j] }; out[i]=s }; return out }
func sumAxis0(A [][]float64) []float64 { m:=len(A); n:=len(A[0]); out:=make([]float64,n); for j:=0;j<n;j++{ s:=0.0; for i:=0;i<m;i++{ s+=A[i][j] }; out[j]=s }; return out }
func outerProduct(a,b []float64) [][]float64 { m:=len(a); n:=len(b); R:=make([][]float64,m); for i:=0;i<m;i++{ R[i]=make([]float64,n); for j:=0;j<n;j++{ R[i][j]=a[i]*b[j] } }; return R }
func elementwiseMul(A,B [][]float64) ([][]float64, error) { m,n:=len(A),len(A[0]); if len(B)!=m||len(B[0])!=n { return nil, fmt.Errorf("shape mismatch") }; R:=make([][]float64,m); for i:=0;i<m;i++{ R[i]=make([]float64,n); for j:=0;j<n;j++{ R[i][j]=A[i][j]*B[i][j] } }; return R, nil }
func reshape2D(data []float64, rows, cols int) [][]float64 { out:=make([][]float64, rows); for i:=0;i<rows;i++{ out[i]=make([]float64, cols); copy(out[i], data[i*cols:(i+1)*cols]) }; return out }
func reshape3D(data []float64, d1, d2, d3 int) [][][]float64 { out:=make([][][]float64,d1); idx:=0; for i:=0;i<d1;i++{ out[i]=make([][]float64,d2); for j:=0;j<d2;j++{ out[i][j]=make([]float64,d3); copy(out[i][j], data[idx:idx+d3]); idx+=d3 } }; return out }
// max moved to internal/ints
