package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDims(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	rows, cols := testMat.Dims()
	assert.Equal(t, 2, rows)
	assert.Equal(t, 2, cols)
	_, err := InitializeMatrix(0, 0)
	assert.Equal(t, "Rows and columns must be greater than 0", err.Error())
}
func TestTranspose(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res, _ := testMat.Tranpose()
	assert.Equal(t, []float32{1, 3}, res.Rows[0].Values)
	assert.Equal(t, []float32{2, 4}, res.Rows[1].Values)
}
func TestString(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.String()
	assert.Equal(t, "| 1 | 2 |\n| 3 | 4 |\n", res)
}
func TestAdd(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 1, 1}}
	var mRow2 = Vector{Values: []float32{3, 4, 6}}
	var mRow3 = Vector{Values: []float32{4, 5, 8}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	var testMat1 = DenseMatrix{Rows: []*Vector{&mRow2, &mRow3}}
	res, _ := testMat.Add(&testMat1)
	assert.Equal(t, []float32{4, 5, 7}, res.Rows[0].Values)
	assert.Equal(t, []float32{7, 9, 14}, res.Rows[1].Values)
}
func TestAddConstant(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res, _ := testMat.AddConstant(15.0)
	assert.Equal(t, []float32{16, 17}, res.Rows[0].Values)
	assert.Equal(t, []float32{18, 19}, res.Rows[1].Values)
}
func TestMultiplyConstant(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res, _ := testMat.MultiplyConstant(10.0)
	assert.Equal(t, []float32{10, 20}, res.Rows[0].Values)
	assert.Equal(t, []float32{30, 40}, res.Rows[1].Values)
}
func TestMultiply(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	var testMat1 = DenseMatrix{Rows: []*Vector{&mRow1, &mRow1}}
	res, _ := testMat.Multiply(&testMat1)
	assert.Equal(t, []float32{3, 6}, res.Rows[0].Values)
	assert.Equal(t, []float32{7, 14}, res.Rows[1].Values)
}
func TestApplyFunc(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res, _ := testMat.ApplyFunc(func(k float32) float32 {
		return k + 3
	})
	assert.Equal(t, []float32{4, 5}, res.Rows[0].Values)
	assert.Equal(t, []float32{6, 7}, res.Rows[1].Values)
	res2, _ := testMat.ApplyFunc(func(k float32) float32 {
		return -1
	})
	assert.Equal(t, []float32{-1, -1}, res2.Rows[0].Values)
	assert.Equal(t, []float32{-1, -1}, res2.Rows[1].Values)
}
func TestReduceSum(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 2}}
	var mRow2 = Vector{Values: []float32{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res, _ := testMat.ReduceSum(0)
	assert.Equal(t, []float32{3}, res.Rows[0].Values)
	assert.Equal(t, []float32{7}, res.Rows[1].Values)
	res2, _ := testMat.ReduceSum(1)
	assert.Equal(t, []float32{4, 6}, res2.Rows[0].Values)
}
