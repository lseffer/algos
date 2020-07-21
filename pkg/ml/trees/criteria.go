package trees

import (
	"math"

	"github.com/lseffer/algos/pkg/ml"
)

type splitCriteria interface {
	formula(data ml.DataSet) float64
}

// GiniCriteria implements Gini Impurity for classification
type GiniCriteria struct{}

// EntropyCriteria implements information gain for classification
type EntropyCriteria struct{}

// MeanSquaredErrorCriteria implements mean squared error for regression
type MeanSquaredErrorCriteria struct{}

func (c GiniCriteria) formula(data ml.DataSet) float64 {
	classCounter := NewClassCounter(data.Target)
	totalCount := len(data.Target.Rows)
	if totalCount == 0 {
		return 0.0
	}
	score := 0.0
	for _, count := range classCounter {
		classRatio := float64(count) / float64(totalCount)
		score = score + classRatio*classRatio
	}
	return 1.0 - score
}

func (c EntropyCriteria) formula(data ml.DataSet) float64 {
	classCounter := NewClassCounter(data.Target)
	totalCount := len(data.Target.Rows)
	if totalCount == 0 {
		return 0.0
	}
	score := 0.0
	for _, count := range classCounter {
		if count > 0 {
			classRatio := float64(count) / float64(totalCount)
			score = score + classRatio*math.Log2(classRatio)
		}
	}
	return -score
}

func (c MeanSquaredErrorCriteria) formula(data ml.DataSet) (res float64) {
	square := func(val float64) float64 {
		return val * val
	}
	res = 0.0
	rowsLength := len(data.Target.Rows)
	if rowsLength == 0 {
		return
	}
	sum := data.Target.ReduceSum(1).Sum()
	prediction := sum / float64(rowsLength)
	differences := data.Target.AddConstant(-prediction)
	differencesSquared := differences.ApplyFunc(square)
	differencesSquaredSum := differencesSquared.ReduceSum(1).Sum()
	res = differencesSquaredSum / float64(rowsLength)
	return
}
