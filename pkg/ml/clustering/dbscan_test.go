package ml

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/lseffer/algos/pkg/matrix"
	"github.com/stretchr/testify/assert"
)

func TestFitDBSCAN(t *testing.T) {
	testdata := mltest.DataSample()
	model := DBScan{NeighborMin: 1, Epsilon: 10.0}
	model.Fit(testdata.Features)
	assert.Equal(t, map[int]int{0: 1, 1: 1, 2: 1}, model.clusterLabels)
}

func TestFitDBSCANTwoDistinctClusters(t *testing.T) {
	var FRow1 = matrix.Vector{Values: []float64{1, 1, 1}}
	var FRow2 = matrix.Vector{Values: []float64{1.2, 1.3, 1.5}}
	var FRow3 = matrix.Vector{Values: []float64{-1000.4, -1110.7, -1110.8}}
	var FRow4 = matrix.Vector{Values: []float64{50, 60, 60}}
	var FRow5 = matrix.Vector{Values: []float64{60, 70, 70}}
	var FRow6 = matrix.Vector{Values: []float64{60.8, 70, 70}}
	var featuresMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&FRow1, &FRow2, &FRow3, &FRow4, &FRow5, &FRow6}}
	model := DBScan{NeighborMin: 0, Epsilon: 10.0}
	model.Fit(&featuresMat)
	prediction := model.clusterLabels
	assert.Equal(t, map[int]int{0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4}, prediction)
}
