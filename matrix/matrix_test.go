package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDims(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1.5, 1.5}, {1.5, 1.5}}}
	rows, cols := testMat.Dims()
	assert.Equal(t, 2, rows)
	assert.Equal(t, 2, cols)
	_, err := InitializeMatrix(0, 0)
	assert.Equal(t, "Rows and columns must be greater than 0", err.Error())
}
func TestTranspose(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	res, _ := testMat.Tranpose()
	assert.Equal(t, [][]float32{{1, 3}, {2, 4}}, res.Rows)
}
func TestString(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	res := testMat.String()
	assert.Equal(t, "| 1 | 2 |\n| 3 | 4 |\n", res)
}
func TestAdd(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 1, 1}, {1, 1, 1}}}
	var testMat1 = DenseMatrix{Rows: [][]float32{{3, 4, 6}, {4, 5, 8}}}
	res, _ := testMat.Add(&testMat1)
	assert.Equal(t, [][]float32{{4, 5, 7}, {5, 6, 9}}, res.Rows)
}
func TestAddConstant(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	res, _ := testMat.AddConstant(15.0)
	assert.Equal(t, [][]float32{{16, 17}, {18, 19}}, res.Rows)
}
func TestMultiplyConstant(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	res, _ := testMat.MultiplyConstant(10.0)
	assert.Equal(t, [][]float32{{10, 20}, {30, 40}}, res.Rows)
}
func TestMultiply(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	var testMat1 = DenseMatrix{Rows: [][]float32{{1, 2}, {1, 2}}}
	res, _ := testMat.Multiply(&testMat1)
	assert.Equal(t, [][]float32{{3, 6}, {7, 14}}, res.Rows)
}
func TestApplyFunc(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	res, _ := testMat.ApplyFunc(func(k float32) float32 {
		return k + 3
	})
	assert.Equal(t, [][]float32{{4, 5}, {6, 7}}, res.Rows)
	res2, _ := testMat.ApplyFunc(func(k float32) float32 {
		return -1
	})
	assert.Equal(t, [][]float32{{-1, -1}, {-1, -1}}, res2.Rows)
}
func TestReduceSum(t *testing.T) {
	var testMat = DenseMatrix{Rows: [][]float32{{1, 2}, {3, 4}}}
	res, _ := testMat.ReduceSum(0)
	assert.Equal(t, [][]float32{{3}, {7}}, res.Rows)
	res2, _ := testMat.ReduceSum(1)
	assert.Equal(t, [][]float32{{4, 6}}, res2.Rows)
}
