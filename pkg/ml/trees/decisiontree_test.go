package trees

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/stretchr/testify/assert"
)

func TestFitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewDecisionTreeClassifier(1, 1, GiniCriteria{}, GreedySplitFinder{})
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}

func TestFitPredictConcurrent(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewDecisionTreeClassifier(1, 1, GiniCriteria{}, ConcurrentSplitFinder{Jobs: 10, SplitFinder: GreedySplitFinder{}})
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}

func TestFitPredictConcurrentRegression(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewDecisionTreeRegressor(1, 1, ConcurrentSplitFinder{Jobs: 10, SplitFinder: GreedySplitFinder{}})
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}
