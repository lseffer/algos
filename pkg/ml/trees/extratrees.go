package trees

import (
	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

// ExtraTrees struct
type ExtraTrees struct {
	maxDepth           int
	minLeafSize        int
	criteria           splitCriteria
	splitFinder        splitFinder
	predictor          predictor
	numberOfEstimators int
	trees              []DecisionTree
	rootNode           *treeNode
}

// Fit the ExtraTrees algorithm
func (m *ExtraTrees) Fit(data ml.DataSet) {
	m.trees = make([]DecisionTree, m.numberOfEstimators)
	for i := 0; i < m.numberOfEstimators; i++ {
		model := DecisionTree{predictor: m.predictor, maxDepth: m.maxDepth, minLeafSize: m.minLeafSize, criteria: m.criteria, splitFinder: m.splitFinder, useBagging: false, randomSubspace: true}
		model.Fit(data)
		m.trees[i] = model
	}
}

// Predict with ExtraTrees
func (m *ExtraTrees) Predict(X *matrix.DenseMatrix) *matrix.DenseMatrix {
	rows, _ := X.Dims()
	res := matrix.InitializeMatrix(rows, 1)
	for _, tree := range m.trees {
		treePred := tree.Predict(X)
		res, _ = res.Add(treePred)
	}
	res = res.MultiplyConstant(1.0 / float64(len(m.trees)))
	return res
}

// NewExtraTreesClassifier create a new instance of a decision tree classifier.
func NewExtraTreesClassifier(maxDepth, minLeafSize int, criteria splitCriteria, numberOfEstimators int) ExtraTrees {
	return ExtraTrees{predictor: ClassificationPredictor{}, maxDepth: maxDepth, minLeafSize: minLeafSize, criteria: criteria, splitFinder: RandomizedSplitFinder{}, numberOfEstimators: numberOfEstimators}
}

// NewExtraTreesRegressor create a new instance of a decision tree regressor.
func NewExtraTreesRegressor(maxDepth, minLeafSize int, numberOfEstimators int) ExtraTrees {
	return ExtraTrees{predictor: RegressionPredicor{}, maxDepth: maxDepth, minLeafSize: minLeafSize, criteria: MeanSquaredErrorCriteria{}, splitFinder: RandomizedSplitFinder{}, numberOfEstimators: numberOfEstimators}
}
