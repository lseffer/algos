package trees

import (
	"math"

	"github.com/lseffer/algos/pkg/ml"
)

type splitCriteria interface {
	formula(classCounter ml.ClassCounter, totalCount int) (float64, error)
}

// GiniCriteria implements Gini Impurity
type GiniCriteria struct{}

// EntropyCriteria implements information gain
type EntropyCriteria struct{}

func (c GiniCriteria) formula(classCounter ml.ClassCounter, totalCount int) (float64, error) {
	if totalCount == 0 {
		return 0.0, nil
	}
	score := 0.0
	for _, count := range classCounter {
		classRatio := float64(count) / float64(totalCount)
		score = score + classRatio*classRatio
	}
	return 1.0 - score, nil
}

func (c EntropyCriteria) formula(classCounter ml.ClassCounter, totalCount int) (float64, error) {
	if totalCount == 0 {
		return 0.0, nil
	}
	score := 0.0
	for _, count := range classCounter {
		if count > 0 {
			classRatio := float64(count) / float64(totalCount)
			score = score + classRatio*math.Log2(classRatio)
		}
	}
	return -score, nil
}

func scoreClassVector(classVector ml.ClassVector, criteria splitCriteria) (float64, error) {
	rows, _ := classVector.Values.Dims()
	result, err := criteria.formula(classVector.Counter, rows)
	return result, err
}
