package tensor

import (
    "math"

    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/maths/util"
)

// FlashAttention is a simplified attention computation used to decouple
// LocalAI from the mono-repo maths implementation. It performs a basic
// scaled dot-product attention across heads.
func FlashAttention(q, k, v *util.Matrix64, scale float64) *util.Matrix64 {
    if q == nil || k == nil || v == nil {
        return util.NewMatrix64(0, 0)
    }
    // Compute scores = q @ k^T
    scores := util.NewMatrix64(q.Rows, k.Rows)
    for i := 0; i < q.Rows; i++ {
        for j := 0; j < k.Rows; j++ {
            sum := 0.0
            for d := 0; d < q.Cols; d++ {
                sum += q.Data[i*q.Stride+d] * k.Data[j*k.Stride+d]
            }
            scores.Data[i*scores.Stride+j] = sum * scale
        }
    }

    // Softmax over j for each i
    for i := 0; i < scores.Rows; i++ {
        row := scores.Data[i*scores.Stride : i*scores.Stride+scores.Cols]
        max := row[0]
        for _, x := range row {
            if x > max {
                max = x
            }
        }
        sum := 0.0
        for j := 0; j < scores.Cols; j++ {
            row[j] = math.Exp(row[j] - max)
            sum += row[j]
        }
        if sum > 0 {
            inv := 1.0 / sum
            for j := 0; j < scores.Cols; j++ {
                row[j] *= inv
            }
        }
    }

    // Output = scores @ v
    out := util.NewMatrix64(scores.Rows, v.Cols)
    for i := 0; i < scores.Rows; i++ {
        for j := 0; j < v.Cols; j++ {
            sum := 0.0
            for t := 0; t < scores.Cols; t++ {
                sum += scores.Data[i*scores.Stride+t] * v.Data[t*v.Stride+j]
            }
            out.Data[i*out.Stride+j] = sum
        }
    }
    return out
}

