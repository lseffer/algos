package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInitialize(t *testing.T) {
	mat0 := InitializeMatrix(0, 0)
	assert.Equal(t, 0, len(mat0.Rows))
	mat01 := InitializeMatrix(1, 0)
	assert.Equal(t, 1, len(mat01.Rows))
	assert.Equal(t, 0, len(mat01.Rows[0].Values))
	mat := InitializeMatrix(0, 1)
	assert.Equal(t, 0, len(mat.Rows))
	matRows, matCols := mat.Dims()
	assert.Equal(t, 0, matRows)
	assert.Equal(t, 0, matCols)
	mat1 := InitializeMatrix(1, 1)
	assert.Equal(t, 1, len(mat1.Rows))
	assert.Equal(t, 1, len(mat1.Rows[0].Values))
}

func TestDims(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	rows, cols := testMat.Dims()
	assert.Equal(t, 2, rows)
	assert.Equal(t, 2, cols)
}
func TestTranspose(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.Tranpose()
	assert.Equal(t, []float64{1, 3}, res.Rows[0].Values)
	assert.Equal(t, []float64{2, 4}, res.Rows[1].Values)
}
func TestString(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.String()
	assert.Equal(t, "| 1 | 2 |\n| 3 | 4 |\n", res)
}
func TestAdd(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 1, 1}}
	var mRow2 = Vector{Values: []float64{3, 4, 6}}
	var mRow3 = Vector{Values: []float64{4, 5, 8}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	var testMat1 = DenseMatrix{Rows: []*Vector{&mRow2, &mRow3}}
	res, _ := testMat.Add(&testMat1)
	assert.Equal(t, []float64{4, 5, 7}, res.Rows[0].Values)
	assert.Equal(t, []float64{7, 9, 14}, res.Rows[1].Values)
}
func TestAddConstant(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.AddConstant(15.0)
	assert.Equal(t, []float64{16, 17}, res.Rows[0].Values)
	assert.Equal(t, []float64{18, 19}, res.Rows[1].Values)
}
func TestMultiplyConstant(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.MultiplyConstant(10.0)
	assert.Equal(t, []float64{10, 20}, res.Rows[0].Values)
	assert.Equal(t, []float64{30, 40}, res.Rows[1].Values)
}
func TestMultiply(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	var testMat1 = DenseMatrix{Rows: []*Vector{&mRow1, &mRow1}}
	res, _ := testMat.Multiply(&testMat1)
	assert.Equal(t, []float64{3, 6}, res.Rows[0].Values)
	assert.Equal(t, []float64{7, 14}, res.Rows[1].Values)
}
func TestApplyFunc(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.ApplyFunc(func(k float64) float64 {
		return k + 3
	})
	assert.Equal(t, []float64{4, 5}, res.Rows[0].Values)
	assert.Equal(t, []float64{6, 7}, res.Rows[1].Values)
	res2 := testMat.ApplyFunc(func(k float64) float64 {
		return -1
	})
	assert.Equal(t, []float64{-1, -1}, res2.Rows[0].Values)
	assert.Equal(t, []float64{-1, -1}, res2.Rows[1].Values)
}
func TestReduceSum(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := testMat.ReduceSum(0)
	assert.Equal(t, []float64{3, 7}, res.Values)
	res2 := testMat.ReduceSum(1)
	assert.Equal(t, []float64{4, 6}, res2.Values)
}
func TestGetSubset(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	resRows, _ := testMat.GetSubset(0, 0, 0)
	assert.Equal(t, 1, len(resRows.Rows))
	assert.Equal(t, []float64{1, 2}, resRows.Rows[0].Values)
	resRows2, _ := testMat.GetSubset(1, 1, 0)
	assert.Equal(t, 1, len(resRows2.Rows))
	assert.Equal(t, []float64{3, 4}, resRows2.Rows[0].Values)
	resCols, _ := testMat.GetSubset(0, 0, 1)
	assert.Equal(t, 2, len(resCols.Rows))
	assert.Equal(t, []float64{1}, resCols.Rows[0].Values)
	assert.Equal(t, []float64{3}, resCols.Rows[1].Values)
	resCols2, _ := testMat.GetSubset(1, 1, 1)
	assert.Equal(t, 2, len(resCols2.Rows))
	assert.Equal(t, []float64{2}, resCols2.Rows[0].Values)
	assert.Equal(t, []float64{4}, resCols2.Rows[1].Values)
}
func TestGetSubsetErr(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	_, errRows := testMat.GetSubset(-1, 0, 0)
	assert.NotNil(t, errRows)
	_, errCols := testMat.GetSubset(0, -1, 0)
	assert.NotNil(t, errCols)
	_, errAxis := testMat.GetSubset(0, 0, 3)
	assert.NotNil(t, errAxis)
}

func qual0th(rowIndex int, values *Vector) bool {
	return values.Values[0] < 2.0
}
func TestSplit(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	resLeft, resRight := Split(&testMat, qual0th)
	assert.Equal(t, 1, len(resLeft))
	assert.Equal(t, 1, len(resRight))
}
func TestGetSubSetByIndex(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 2}}
	var mRow2 = Vector{Values: []float64{3, 4}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2}}
	res := GetSubSetByIndex(&testMat, []int{0, 1})
	assert.Equal(t, &testMat, res)
	res2 := GetSubSetByIndex(&testMat, []int{0})
	assert.Equal(t, &DenseMatrix{Rows: []*Vector{&mRow1}}, res2)
}

func TestMinMax(t *testing.T) {
	var mRow1 = Vector{Values: []float64{1, 1, 11}}
	var mRow2 = Vector{Values: []float64{10, 4, 6}}
	var mRow3 = Vector{Values: []float64{4, 5, 8}}
	var testMat = DenseMatrix{Rows: []*Vector{&mRow1, &mRow2, &mRow3}}
	minRes, maxRes := testMat.MinMax()
	assert.Equal(t, []float64{10, 5, 11}, maxRes)
	assert.Equal(t, []float64{1, 1, 6}, minRes)
}
