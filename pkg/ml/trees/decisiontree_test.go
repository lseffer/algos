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

func TestFitPredictBag(t *testing.T) {
	// Bagging is not guaranteed to always give perfect results, so we only test it compiles and runs.
	testdata := mltest.DataSample()
	model := NewDecisionTreeClassifier(1, 1, GiniCriteria{}, GreedySplitFinder{})
	model.useBagging = true
	model.Fit(testdata)
	model.Predict(testdata.Features)
}

func TestFitPredictConcurrent(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewDecisionTreeClassifier(1, 1, GiniCriteria{}, ConcurrentSplitFinder{jobs: 10, s: GreedySplitFinder{}})
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}

func TestFitPredictConcurrentRegression(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewDecisionTreeRegressor(1, 1, ConcurrentSplitFinder{jobs: 10, s: GreedySplitFinder{}})
	model.Fit(testdata)
	prediction := model.Predict(testdata.Features)
	assert.Equal(t, testdata.Target, prediction)
}
