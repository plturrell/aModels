package conv

import (
    "reflect"
    "testing"
)

func TestStandardConv2D_Basic2x2Kernel(t *testing.T) {
    input := [][]float64{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    }
    kernel := [][]float64{
        {1, 0},
        {0, 1},
    }
    got, err := standardConv2D(input, kernel, 1)
    if err != nil { t.Fatalf("unexpected error: %v", err) }
    want := [][]float64{
        {7, 9, 11},
        {15, 17, 19},
        {23, 25, 27},
    }
    if !reflect.DeepEqual(got, want) {
        t.Fatalf("conv result mismatch:\n got=%v\nwant=%v", got, want)
    }
}

func TestStandardConv2D_Stride2(t *testing.T) {
    input := [][]float64{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    }
    kernel := [][]float64{
        {1, 0},
        {0, 1},
    }
    got, err := standardConv2D(input, kernel, 2)
    if err != nil { t.Fatalf("unexpected error: %v", err) }
    want := [][]float64{
        {7, 11},
        {23, 27},
    }
    if !reflect.DeepEqual(got, want) {
        t.Fatalf("conv stride2 mismatch:\n got=%v\nwant=%v", got, want)
    }
}

func TestStandardConv2D_KernelLargerThanInput(t *testing.T) {
    input := [][]float64{{1, 2}, {3, 4}}
    kernel := [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    if _, err := standardConv2D(input, kernel, 1); err == nil {
        t.Fatalf("expected error for kernel larger than input")
    }
}

func TestStandardConv2D_NonSquareKernel(t *testing.T) {
    input := [][]float64{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    }
    kernel := [][]float64{
        {1, 2, 3},
        {4, 5, 6},
    }
    got, err := standardConv2D(input, kernel, 1)
    if err != nil { t.Fatalf("unexpected error: %v", err) }
    want := [][]float64{
        {106, 127},
        {190, 211},
        {274, 295},
    }
    if !reflect.DeepEqual(got, want) {
        t.Fatalf("conv non-square mismatch:\n got=%v\nwant=%v", got, want)
    }
}

func TestWinogradConv2D_Compatibility(t *testing.T) {
    input := [][]float64{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    }
    kernel := [][]float64{
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1},
    }
    sref, err := standardConv2D(input, kernel, 1)
    if err != nil { t.Fatalf("unexpected error: %v", err) }
    got, err := WinogradConv2D(input, kernel, 1)
    if err != nil { t.Fatalf("unexpected error: %v", err) }
    if !reflect.DeepEqual(got, sref) {
        t.Fatalf("winograd wrapper mismatch vs standard:\n got=%v\nref=%v", got, sref)
    }
}

