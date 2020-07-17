package mltest

import (
	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

// DataSample generate a small dataset
func DataSample() ml.DataSet {
	var FRow1 = matrix.Vector{Values: []float64{1, 1, 1}}
	var FRow2 = matrix.Vector{Values: []float64{3, 4, 6}}
	var FRow3 = matrix.Vector{Values: []float64{4, 5, 8}}
	var featuresMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&FRow1, &FRow2, &FRow3}}
	var TRow1 = matrix.Vector{Values: []float64{0}}
	var TRow2 = matrix.Vector{Values: []float64{0}}
	var TRow3 = matrix.Vector{Values: []float64{1}}
	var targetMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&TRow1, &TRow2, &TRow3}}
	return ml.DataSet{Features: &featuresMat, Target: &targetMat}
}

// DataSampleOneClass generate a small dataset
func DataSampleOneClass() ml.DataSet {
	var FRow1 = matrix.Vector{Values: []float64{1, 1, 1}}
	var FRow2 = matrix.Vector{Values: []float64{3, 4, 6}}
	var FRow3 = matrix.Vector{Values: []float64{4, 5, 8}}
	var featuresMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&FRow1, &FRow2, &FRow3}}
	var TRow1 = matrix.Vector{Values: []float64{0}}
	var TRow2 = matrix.Vector{Values: []float64{0}}
	var TRow3 = matrix.Vector{Values: []float64{0}}
	var targetMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&TRow1, &TRow2, &TRow3}}
	return ml.DataSet{Features: &featuresMat, Target: &targetMat}
}

// DataSampleEvenSpread generate a small dataset
func DataSampleEvenSpread() ml.DataSet {
	var FRow1 = matrix.Vector{Values: []float64{1, 1, 1}}
	var FRow2 = matrix.Vector{Values: []float64{3, 4, 6}}
	var FRow3 = matrix.Vector{Values: []float64{4, 5, 8}}
	var FRow4 = matrix.Vector{Values: []float64{6, 7, 12}}
	var featuresMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&FRow1, &FRow2, &FRow3, &FRow4}}
	var TRow1 = matrix.Vector{Values: []float64{0}}
	var TRow2 = matrix.Vector{Values: []float64{0}}
	var TRow3 = matrix.Vector{Values: []float64{1}}
	var TRow4 = matrix.Vector{Values: []float64{1}}
	var targetMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&TRow1, &TRow2, &TRow3, &TRow4}}
	return ml.DataSet{Features: &featuresMat, Target: &targetMat}
}
