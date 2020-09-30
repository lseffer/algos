package trees

import (
	"math"
	"math/rand"

	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

// DecisionTree ml model
// base struct for decision trees
type DecisionTree struct {
	maxDepth    int
	minLeafSize int
	rootNode    *treeNode
	criteria    splitCriteria
	splitFinder splitFinder
	predictor   predictor
	useBagging  bool
}

// Fit the decision tree
func (m *DecisionTree) Fit(data ml.DataSet) {
	m.rootNode = &treeNode{depth: 0}
	s := make(treeStack, 0)
	s = s.Push(m.rootNode)
	m.buildTree(data, s, m.useBagging)
}

func (m *DecisionTree) buildTree(data ml.DataSet, s treeStack, bagging bool) {
	var left, right, current *treeNode
	var splitRes splitResults
	var dataForSplit ml.DataSet
	rows, cols := data.Features.Dims()
	heuristicBagSize := int(math.Floor(math.Sqrt(float64(cols))))
	if s.Size() <= 0 {
		return
	}
	if rows < m.minLeafSize {
		return
	}
	s, current = s.Pop()
	current.prediction = m.predictor.predict(data)
	if current.depth+1 > m.maxDepth {
		return
	}
	if bagging {
		colIndices := make([]int, heuristicBagSize)
		for idx := range colIndices {
			colIndices[idx] = rand.Intn(cols)
		}
		dataForSplit = ml.DataSet{Features: matrix.GetSubSetByColIndex(data.Features, colIndices), Target: data.Target}
	} else {
		dataForSplit = data
	}
	splitRes = m.splitFinder.algorithm(dataForSplit, m.criteria, 0, rows)
	current.score = splitRes.score
	current.colIndex = splitRes.colIndex
	current.splitValue = splitRes.splitValue
	left = &treeNode{depth: current.depth + 1}
	right = &treeNode{depth: current.depth + 1}
	current.left = left
	current.right = right
	s = s.Push(left)
	m.buildTree(splitRes.leftData, s, m.useBagging)
	s = s.Push(right)
	m.buildTree(splitRes.rightData, s, m.useBagging)
}

func (m *DecisionTree) predictRow(current *treeNode, row *matrix.Vector, prediction ml.TargetValue) ml.TargetValue {
	if current.left == nil && current.right == nil {
		return current.prediction
	}
	prediction = current.prediction
	if row.Values[current.colIndex] < current.splitValue {
		current = current.left
		return m.predictRow(current, row, prediction)
	}
	current = current.right
	return m.predictRow(current, row, prediction)
}

// Predict on data using the classifier
func (m *DecisionTree) Predict(X *matrix.DenseMatrix) *matrix.DenseMatrix {
	initPrediction := ml.TargetValue(0.0)
	rows, _ := X.Dims()
	predicted := matrix.InitializeMatrix(rows, 1)
	for rowIndex, rowVector := range X.Rows {
		prediction := m.predictRow(m.rootNode, rowVector, initPrediction)
		predicted.Rows[rowIndex].Values[0] = float64(prediction)
	}
	return predicted
}

// NewDecisionTreeClassifier create a new instance of a decision tree classifier.
func NewDecisionTreeClassifier(maxDepth, minLeafSize int, criteria splitCriteria, splitFinder splitFinder) DecisionTree {
	return DecisionTree{predictor: ClassificationPredictor{}, maxDepth: maxDepth, minLeafSize: minLeafSize, criteria: criteria, splitFinder: splitFinder}
}

// NewDecisionTreeRegressor create a new instance of a decision tree regressor.
func NewDecisionTreeRegressor(maxDepth, minLeafSize int, splitFinder splitFinder) DecisionTree {
	return DecisionTree{predictor: RegressionPredicor{}, maxDepth: maxDepth, minLeafSize: minLeafSize, criteria: MeanSquaredErrorCriteria{}, splitFinder: splitFinder}
}
