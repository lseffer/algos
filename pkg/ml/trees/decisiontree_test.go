package trees

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/stretchr/testify/assert"
)

func TestFitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := DecisionTree{predictor: ClassificationPredictor{}, maxDepth: 1, minLeafSize: 1, criteria: GiniCriteria{}, splitFinder: GreedySplitFinder{}}
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}

func TestFitPredictConcurrent(t *testing.T) {
	testdata := mltest.DataSample()
	model := DecisionTree{predictor: ClassificationPredictor{}, maxDepth: 1, minLeafSize: 1, criteria: GiniCriteria{}, splitFinder: ConcurrentSplitFinder{jobs: 10, s: GreedySplitFinder{}}}
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}
