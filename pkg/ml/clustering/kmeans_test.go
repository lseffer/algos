package ml

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/lseffer/algos/pkg/matrix"
	"github.com/stretchr/testify/assert"
)

func TestFitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := KMeans{ClusterCount: 1, MaxIterations: 25, Tolerance: 0.01}
	model.Fit(testdata.Features)
	prediction, _ := model.Predict(testdata.Features)
	assert.Equal(t, []float64{0, 0, 0}, prediction.Values)
}

func TestFitPredictTwoDistinctClusters(t *testing.T) {
	var FRow1 = matrix.Vector{Values: []float64{1, 1, 1}}
	var FRow2 = matrix.Vector{Values: []float64{1.2, 1.3, 1.5}}
	var FRow3 = matrix.Vector{Values: []float64{1.4, 1.7, 1.8}}
	var FRow4 = matrix.Vector{Values: []float64{5, 6, 6}}
	var FRow5 = matrix.Vector{Values: []float64{6, 7, 7}}
	var FRow6 = matrix.Vector{Values: []float64{6.8, 7, 7}}
	var featuresMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&FRow1, &FRow2, &FRow3, &FRow4, &FRow5, &FRow6}}
	model := KMeans{ClusterCount: 2, MaxIterations: 25, Tolerance: 0.01}
	model.Fit(&featuresMat)
	prediction, _ := model.Predict(&featuresMat)
	firstCluster := prediction.Values[0]
	secondCluster := prediction.Values[3]
	assert.Equal(t, firstCluster, prediction.Values[1])
	assert.Equal(t, firstCluster, prediction.Values[2])
	assert.Equal(t, secondCluster, prediction.Values[4])
	assert.Equal(t, secondCluster, prediction.Values[5])
}
