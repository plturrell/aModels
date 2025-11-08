package util

// Q16FromFloat converts float64 to Q16.16 fixed-point (int32) with rounding.
// Range is approximately [-32768, 32767.99998]. Values outside are clamped.
func Q16FromFloat(x float64) int32 {
	if x > 32767.99998 {
		x = 32767.99998
	} else if x < -32768.0 {
		x = -32768.0
	}
	v := x * 65536.0
	if v >= 0 {
		return int32(v + 0.5)
	}
	return int32(v - 0.5)
}

// Q16ToFloat converts Q16.16 fixed-point (int32) to float64.
func Q16ToFloat(q int32) float64 { return float64(q) / 65536.0 }
