package ml

import (
	"errors"
	"math/rand"
	"time"

	"github.com/lseffer/algos/pkg/matrix"
)

// TargetValue a single value of a target matrix.
type TargetValue float64

// DataSet is a collection of data objects for convenient downstream usage.
type DataSet struct {
	Features *matrix.DenseMatrix
	Target   *matrix.DenseMatrix
}

// NewDataSetNoTarget create a new dataset from only one matrix
func NewDataSetNoTarget(X *matrix.DenseMatrix) (result *DataSet) {
	result = &DataSet{Features: X}
	return result
}

// NewDataSet create a new dataset from a matrix and a vector
func NewDataSet(X *matrix.DenseMatrix, y *matrix.DenseMatrix) (result DataSet, err error) {
	xRows, _ := X.Dims()
	yRows, yCols := y.Dims()
	if yCols != 1 {
		return DataSet{}, errors.New("y must be a 1 dimensional matrix, i.e. a column vector")
	}
	if yRows != xRows {
		return DataSet{}, errors.New("X and y must have the same number of rows")
	}
	result = DataSet{Features: X, Target: y}
	return
}

func randomQualifierFactory(trainSize float64) (res matrix.Qualifier) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	res = func(rowIndex int, values *matrix.Vector) bool {
		return r.Float64() < trainSize
	}
	return
}

// TrainTestSplit split a dataset randomly into training and testing using a random fraction
func TrainTestSplit(dataset DataSet, trainSize float64) (train DataSet, test DataSet) {
	trainIdx, testIdx := matrix.Split(dataset.Features, randomQualifierFactory(trainSize))
	train.Features = matrix.GetSubSetByIndex(dataset.Features, trainIdx)
	train.Target = matrix.GetSubSetByIndex(dataset.Target, trainIdx)
	test.Features = matrix.GetSubSetByIndex(dataset.Features, testIdx)
	test.Target = matrix.GetSubSetByIndex(dataset.Target, testIdx)
	return
}
