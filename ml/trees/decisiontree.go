package trees

import (
	"algos/matrix"
	"algos/ml"
)

type treeNode struct {
	score         float64
	colIndex      int
	splitValue    float64
	depth         int
	majorityClass ml.ClassValue
	left          *treeNode
	right         *treeNode
}

// DecisionTreeClassifier ml model
type DecisionTreeClassifier struct {
	maxDepth    int
	minLeafSize int
	criteria    splitCriteria
	classes     []float64
	rootNode    *treeNode
	splitFinder splitFinder
}

// Fit the decision tree classifier. Assume the target is the last column to the right of the matrix
func (m *DecisionTreeClassifier) Fit(X *matrix.DenseMatrix) {
	m.rootNode = &treeNode{depth: 0}
	s := make(treeStack, 0)
	s = s.Push(m.rootNode)
	m.buildTree(X, s)
}

func (m *DecisionTreeClassifier) buildTree(X *matrix.DenseMatrix, s treeStack) {
	var left, right, current *treeNode
	var splitRes splitResults
	var err error
	var classVec ml.ClassVector
	if s.Size() <= 0 {
		return
	}
	if len(X.Rows) < m.minLeafSize {
		return
	}
	s, current = s.Pop()
	if current.depth+1 > m.maxDepth {
		return
	}
	classVec, err = ml.NewClassVector(X)
	splitRes, err = m.splitFinder.algorithm(X, m.criteria)
	if err != nil {
		return
	}
	current.score = splitRes.score
	current.colIndex = splitRes.colIndex
	current.splitValue = splitRes.splitValue
	current.majorityClass = classVec.MajorityClass
	left = &treeNode{depth: current.depth + 1}
	right = &treeNode{depth: current.depth + 1}
	current.left = left
	current.right = right
	s = s.Push(left)
	m.buildTree(splitRes.leftData, s)
	s = s.Push(right)
	m.buildTree(splitRes.rightData, s)
}

func (m *DecisionTreeClassifier) predictRow(current *treeNode, row *matrix.Vector, prediction ml.ClassValue) ml.ClassValue {
	if current.left == nil && current.right == nil {
		return prediction
	}
	prediction = current.majorityClass
	if row.Values[current.colIndex] < current.splitValue {
		current = current.left
		return m.predictRow(current, row, prediction)
	}
	current = current.right
	return m.predictRow(current, row, prediction)
}

// Predict on data using the classifier
func (m *DecisionTreeClassifier) Predict(X *matrix.DenseMatrix) (*matrix.DenseMatrix, error) {
	rows, _ := X.Dims()
	predicted, err := matrix.InitializeMatrix(rows, 1)
	for rowIndex, rowVector := range X.Rows {
		prediction := m.predictRow(m.rootNode, rowVector, 0.0)
		predicted.Rows[rowIndex].Values[0] = float64(prediction)
	}
	return predicted, err
}
