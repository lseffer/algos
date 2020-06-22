package trees

import (
	"algos/ml"
	"math"
)

type splitCriteria interface {
	formula(classCounter ml.ClassCounter, totalCount int) (float64, error)
}

type giniCriteria struct{}
type entropyCriteria struct{}

func (c giniCriteria) formula(classCounter ml.ClassCounter, totalCount int) (float64, error) {
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

func (c entropyCriteria) formula(classCounter ml.ClassCounter, totalCount int) (float64, error) {
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
