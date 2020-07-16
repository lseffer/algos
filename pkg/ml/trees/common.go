package trees

import (
	"sort"

	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

type treeNode struct {
	score      float64
	colIndex   int
	splitValue float64
	depth      int
	prediction ml.TargetValue
	left       *treeNode
	right      *treeNode
}

type treeStack []*treeNode

func (s treeStack) Size() int {
	return len(s)
}

func (s treeStack) Push(v *treeNode) treeStack {
	return append(s, v)
}

func (s treeStack) Pop() (treeStack, *treeNode) {

	l := len(s)
	if l <= 0 {
		return nil, nil
	}
	return s[:l-1], s[l-1]
}

// Predictor interface for making predictions based on tree leafs. I.e. from the dataset itself.
type predictor interface {
	predict(ml.DataSet) ml.TargetValue
}

// ClassificationPredictor uses the majority class of the target data as a prediction
type ClassificationPredictor struct{}

// RegressionPredicor uses the average value of the target data as a prediction
type RegressionPredicor struct{}

func (c ClassificationPredictor) predict(data ml.DataSet) ml.TargetValue {
	classCounter := NewClassCounter(data.Target)
	return GetMajorityClass(classCounter)
}

func (c RegressionPredicor) predict(data ml.DataSet) (prediction ml.TargetValue) {
	sum := data.Target.ReduceSum(1).Values[0]
	prediction = ml.TargetValue(sum / float64(len(data.Target.Rows)))
	return
}

// ClassCounter is a map that counts class occurences
type ClassCounter map[ml.TargetValue]int

// NewClassCounter count all unique values of 1-d matrix
func NewClassCounter(X *matrix.DenseMatrix) ClassCounter {
	var classVal ml.TargetValue
	result := make(map[ml.TargetValue]int)
	for _, rowVector := range X.Rows {
		classVal = ml.TargetValue(rowVector.Values[0])
		if _, ok := result[classVal]; ok {
			result[classVal]++
		} else {
			result[classVal] = 1
		}
	}
	return result
}

// GetMajorityClass gets the class with the largest count from the classcounter map.
// Keys will be sorted in ascending order before finding the majority class.
func GetMajorityClass(classCounter ClassCounter) (res ml.TargetValue) {
	keys := make([]float64, 0)
	for k := range classCounter {
		keys = append(keys, float64(k))
	}
	sort.Float64s(keys)
	var val int
	maxCount := 0
	for _, key := range keys {
		val = classCounter[ml.TargetValue(key)]
		if val > maxCount {
			res = ml.TargetValue(key)
			maxCount = val
		}
	}
	return
}
