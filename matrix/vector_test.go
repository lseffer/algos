package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEuclideanDistance1(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1}}
	var mRow2 = Vector{Values: []float32{3}}
	res, _ := mRow1.EuclideanDistance(&mRow2)
	assert.Equal(t, float32(2.0), res)
}
func TestEuclideanDistance2(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 1, 1, 1}}
	var mRow2 = Vector{Values: []float32{3, 3, 3, 3}}
	res, _ := mRow1.EuclideanDistance(&mRow2)
	assert.Equal(t, float32(4.0), res)
}

func TestSum(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 1, 1, 1}}
	res := mRow1.Sum()
	assert.Equal(t, float32(4.0), res)
}
func TestAddVec(t *testing.T) {
	var mRow1 = Vector{Values: []float32{1, 1, 1, 1}}
	var mRow2 = Vector{Values: []float32{3, 3, 3, 3}}
	res, _ := mRow1.Add(&mRow2)
	assert.Equal(t, []float32{4, 4, 4, 4}, res.Values)
}
