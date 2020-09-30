package trees

import (
	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

// RandomForest struct
type RandomForest struct {
	maxDepth           int
	minLeafSize        int
	criteria           splitCriteria
	splitFinder        splitFinder
	predictor          predictor
	numberOfEstimators int
	trees              []DecisionTree
	rootNode           *treeNode
}

// Fit the randomforest algorithm
func (m *RandomForest) Fit(data ml.DataSet) {
	m.trees = make([]DecisionTree, m.numberOfEstimators)
	for i := 0; i < m.numberOfEstimators; i++ {
		model := DecisionTree{predictor: m.predictor, maxDepth: m.maxDepth, minLeafSize: m.minLeafSize, criteria: m.criteria, splitFinder: m.splitFinder, useBagging: true}
		model.Fit(data)
		m.trees[i] = model
	}
}

// Predict with randomforest
func (m *RandomForest) Predict(X *matrix.DenseMatrix) *matrix.DenseMatrix {
	rows, _ := X.Dims()
	res := matrix.InitializeMatrix(rows, 1)
	for _, tree := range m.trees {
		treePred := tree.Predict(X)
		res, _ = res.Add(treePred)
	}
	res = res.MultiplyConstant(1.0 / float64(len(m.trees)))
	return res
}

// NewRandomForestClassifier create a new instance of a decision tree classifier.
func NewRandomForestClassifier(maxDepth, minLeafSize int, criteria splitCriteria, splitFinder splitFinder, numberOfEstimators int) RandomForest {
	return RandomForest{predictor: ClassificationPredictor{}, maxDepth: maxDepth, minLeafSize: minLeafSize, criteria: criteria, splitFinder: splitFinder, numberOfEstimators: numberOfEstimators}
}

// NewRandomForestRegressor create a new instance of a decision tree regressor.
func NewRandomForestRegressor(maxDepth, minLeafSize int, splitFinder splitFinder, numberOfEstimators int) RandomForest {
	return RandomForest{predictor: RegressionPredicor{}, maxDepth: maxDepth, minLeafSize: minLeafSize, criteria: MeanSquaredErrorCriteria{}, splitFinder: splitFinder, numberOfEstimators: numberOfEstimators}
}
