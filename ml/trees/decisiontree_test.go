package trees

import (
	"algos/matrix"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFit(t *testing.T) {
	var mRow1 = matrix.Vector{Values: []float64{1, 2, 4, 0}}
	var mRow2 = matrix.Vector{Values: []float64{3, 4, 4, 0}}
	var mRow3 = matrix.Vector{Values: []float64{3, 4, 4, 0}}
	var mRow4 = matrix.Vector{Values: []float64{3, 4, 10, 1}}
	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2, &mRow3, &mRow4}}
	var mCol1 = matrix.Vector{Values: []float64{0}}
	var mCol2 = matrix.Vector{Values: []float64{0}}
	var mCol3 = matrix.Vector{Values: []float64{0}}
	var mCol4 = matrix.Vector{Values: []float64{1}}
	var testVec = matrix.DenseMatrix{Rows: []*matrix.Vector{&mCol1, &mCol2, &mCol3, &mCol4}}
	model := DecisionTreeClassifier{maxDepth: 1, minLeafSize: 1, criteria: giniCriteria{}, splitFinder: greedySplitFinder{}}
	model.Fit(&testMat)
	prediction, _ := model.Predict(&testMat)
	assert.Equal(t, testVec, *prediction)
}
