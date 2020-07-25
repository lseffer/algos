package trees

import (
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
}

// Fit the decision tree
func (m *DecisionTree) Fit(data ml.DataSet) {
	m.rootNode = &treeNode{depth: 0}
	s := make(treeStack, 0)
	s = s.Push(m.rootNode)
	m.buildTree(data, s)
}

func (m *DecisionTree) buildTree(data ml.DataSet, s treeStack) {
	var left, right, current *treeNode
	var splitRes splitResults
	rows, _ := data.Features.Dims()
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
	splitRes = m.splitFinder.algorithm(data, m.criteria, 0, rows)
	current.score = splitRes.score
	current.colIndex = splitRes.colIndex
	current.splitValue = splitRes.splitValue
	left = &treeNode{depth: current.depth + 1}
	right = &treeNode{depth: current.depth + 1}
	current.left = left
	current.right = right
	s = s.Push(left)
	m.buildTree(splitRes.leftData, s)
	s = s.Push(right)
	m.buildTree(splitRes.rightData, s)
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
